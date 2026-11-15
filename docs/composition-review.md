# Prüfung der Box-Komposition – 25./26. Oktober 2026

Geprüft wurden Editor und Gradio-Anbindung, Datenvalidierung und Metadaten, Masken und Prompt-Encoding,
Sampler, Attention, Refiner und das Aufräumen nach abgeschlossenen oder unterbrochenen Aufgaben.
Stand 26. Oktober: 83 Python-Tests bestanden, einschließlich angrenzender Enhance-, Cache- und Performance-Tests.
Die Python-Syntaxprüfung aller geänderten Python-Dateien bestand ebenfalls. Der erweiterte Browser-Test
bestand am 25. Oktober; seitdem wurde die Oberfläche nicht geändert.

## Behobene Befunde

| Befund | Änderung | Nachweis |
| --- | --- | --- |
| Identische Box-Prompts erzeugten wiederholte Berechnungen. | Gewichtete Masken identischer Text-Workloads zusammenführen; Hintergrund aus allen ursprünglichen Boxen erhalten. | Sechs Boxen mit zwei unterschiedlichen Prompts benötigen drei statt sieben Cross-Attention-Aufrufe pro Ebene, einschließlich des globalen Prompts. Vergleich mit Einzelberechnung in FP32, FP16 und BF16, auch bei überlappenden Boxen und verschiedenen Stärken. |
| Jede Addition legte einen weiteren vollständigen Feature-Tensor an. | Den privaten Ergebnis-Akkumulator wiederverwenden. | CPU-Profiler: sechs Additionen benötigen keine zusätzlichen Ergebnistensoren. Eingaben bleiben unverändert; wiederholte Aufrufe liefern passende Ergebnisse. |
| Skip, Stop und Fehler umgingen das Freigeben von Conditioning im Aufgabenstapel. | Aufräumen in `finally`, auch vor dem nächsten Bild. | Regressionstest führt die tatsächliche Worker-Schleife für Erfolg, Skip, Stop und Fehler aus. |
| Sub-Quadratic Attention verwendete eine von PyTorch beanstandete Listen-Indizierung. | Slices als Tupel übergeben. | Tatsächliche Attention-Implementierung gegen SDPA geprüft. |
| Die gleiche räumliche Query wurde für jeden Box-Prompt erneut projiziert. | Standard-Cross-Attention berechnet die Query-Projektion einmal je Block-Aufruf. | Bei sechs unterschiedlichen Boxen eine statt sieben Projektionen. Bitgleiche CPU-Ergebnisse in FP32/FP16/BF16, Batch 1/2 und bei wechselnden Eingaben. |

Die Zusammenführung vorhandener Modell-Hooks mit der Box-Steuerung wurde bereits zuvor korrigiert.
Der verbundene Sampler/SpatialTransformer-Test prüft diesen Pfad weiterhin.

## Messgrenzen

Die Allokationsmessung isoliert das Zusammenführen bereits berechneter Attention-Ergebnisse:
Raster 28 × 36, 1280 Kanäle, Batch 1, FP16, sechs Regionen. Ein Feature-Tensor belegt 2.4609375 MiB.
Die vorherigen sechs Additionen allokierten zusammen 14.765625 MiB; die aktuellen Additionen 0 zusätzliche
Bytes. Das Ergebnis dieses isolierten Vergleichs ist bitgleich. Dies beschreibt kumulative Allokationen
des Mischschritts, keine VRAM-Spitze und keine Beschleunigung der gesamten Bildgenerierung.
Beim Gruppieren identischer Prompts sind kleine Rundungsunterschiede möglich.

Die Query-Wiederverwendung wurde zusätzlich an einem vollständigen Transformer-Block mit zufälligen
Gewichten gemessen: Raster 28 × 36, 640 Kanäle, 10 Heads, Textdimension 2048, jeweils 77 Text-Tokens,
sechs regionale Kontexte, Batch 1, FP32, PyTorch 2.10.0 CPU, zwei Threads. Nach dem Aufwärmen folgten
16 Messpaare mit wechselnder Reihenfolge:

| Variante | Median | Interquartilsabstand |
| --- | ---: | ---: |
| Unabhängige Query-Projektionen | 250,3 ms | 47,1 ms |
| Geteilte Query-Projektion | 207,4 ms | 32,6 ms |

Der Median der gepaarten Zeitverhältnisse lag bei 0,837. Die Streuung ist deutlich; dieser CPU-Befund
ist keine zugesicherte Beschleunigung und lässt sich nicht auf eine vollständige SDXL-Generierung oder
CUDA übertragen. Gesichert ist die Reduktion der Query-Projektionen. Die Query bleibt nur für diesen
Block-Aufruf erhalten; es entsteht kein persistenter Cache je Ebene oder Sampling-Schritt.

## Weitere Prüfabdeckung

- PyTorch-SDPA, Basic, Split und Sub-Quadratic Attention, einschließlich Chunking bei kleinem simuliertem Speicherbudget.
- Unterschiedliche Textlängen, Überlappungen, Bild-Batches, Hoch- und Querformat einschließlich 1152 × 896 und ungerader Maskenraster.
- PAG, Token Merging, Image-Prompt-Ersetzung und DeepCache-Bypass; regionale Embeddings und neue Caches beim Refiner-Wechsel.
- Query-Wiederverwendung mit manuellem Gewichts-Casting; Rückfall auf unabhängige Projektionen bei Attention-Ersetzungen, Query-Patches und lokalen/globalen Modul-Hooks. CPU-Dynamo-Erfassung mit einem Eager-Backend und tatsächlich erfassten Graphen, jeweils mit und ohne Komposition.
- Browser: Texteingabe, Auswahl, Zeichnen, Bewegen, Skalieren, Metadaten-Restore, Grenzen, Sperren während der Generierung und wiederholtes Umschalten danach.

Die Browserprüfung simuliert die Gradio-Textbox-Brücke. Die Transformer-Prüfung verwendet echte
PyTorch-Operationen mit kleinen zufälligen Gewichten; der GPU-Modell-Loader wird dabei nicht gestartet.

## Noch offen

Ein SDXL-Vergleich auf der Ziel-GPU muss Bildzusammenhang, Positionstreue, Laufzeit und maximal belegten
VRAM bestätigen. Ebenfalls offen sind native CUDA-/xFormers-Kernels, tatsächliche `torch.compile`-
Ausführung auf CUDA und ein vollständiger Durchlauf im laufenden Fooocus-Server.

Weiteres Potenzial liegt im Zwischenspeichern der Text-Key/Value-Projektionen je Attention-Ebene.
Das benötigt Messungen auf der Ziel-GPU, weil zusätzliche Key/Value-Caches den Speicherverbrauch deutlich
erhöhen können. Dieser Cache wurde nicht eingebaut. Die Query-Wiederverwendung ist mit den beschriebenen
Hook-Fallbacks umgesetzt; ihr Effekt auf Laufzeit und Speicher der Ziel-GPU bleibt zu messen.
