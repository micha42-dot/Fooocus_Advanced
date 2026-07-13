import ldm_patched.modules.args_parser as args_parser

args_parser.parser.add_argument("--share", action='store_true', help="Set whether to share on Gradio.")

args_parser.parser.add_argument("--preset", type=str, default=None, help="Apply specified UI preset.")
args_parser.parser.add_argument("--disable-preset-selection", action='store_true',
                                help="Disables preset selection in Gradio.")

args_parser.parser.add_argument("--language", type=str, default='default',
                                help="Translate UI using json files in [language] folder. "
                                  "For example, [--language example] will use [language/example.json] for translation.")

# For example, https://github.com/lllyasviel/Fooocus/issues/849
args_parser.parser.add_argument("--disable-offload-from-vram", action="store_true",
                                help="Force loading models to vram when the unload can be avoided. "
                                  "Some Mac users may need this.")

args_parser.parser.add_argument("--theme", type=str, help="launches the UI with light or dark theme", default=None)
args_parser.parser.add_argument("--disable-image-log", action='store_true',
                                help="Prevent writing images and logs to the outputs folder.")

args_parser.parser.add_argument("--disable-analytics", action='store_true',
                                help="Disables analytics for Gradio.")

args_parser.parser.add_argument("--disable-metadata", action='store_true',
                                help="Disables saving metadata to images.")

args_parser.parser.add_argument("--disable-preset-download", action='store_true',
                                help="Disables downloading models for presets", default=False)

args_parser.parser.add_argument("--disable-model-download", action='store_true',
                                help="Disables all automatic model downloads at startup and on preset changes",
                                default=False)

args_parser.parser.add_argument("--disable-enhance-output-sorting", action='store_true',
                                help="Disables enhance output sorting for final image gallery.")

args_parser.parser.add_argument("--enable-auto-describe-image", action='store_true',
                                help="Enables automatic description of uov and enhance image when prompt is empty", default=False)

args_parser.parser.add_argument("--attention-backend", choices=['auto', 'legacy', 'pytorch', 'xformers'],
                                default='auto',
                                help="Select the attention backend. Auto benchmarks PyTorch SDPA and xFormers at startup.")

args_parser.parser.add_argument("--torch-compile", action='store_true',
                                help="Compile the SDXL UNet on first use. The first generation is slower.")
args_parser.parser.add_argument("--torch-compile-mode",
                                choices=['default', 'reduce-overhead', 'max-autotune'],
                                default='reduce-overhead',
                                help="Optimization mode used by torch.compile.")
args_parser.parser.add_argument("--torch-compile-profile", choices=['dynamic', 'resolution'],
                                default='resolution',
                                help="Compile one flexible graph or optimized graphs per latent resolution.")
args_parser.parser.add_argument("--torch-compile-max-profiles", type=int, default=3,
                                help="Maximum number of resolution-specific compiled UNet graphs.")

args_parser.parser.add_argument("--unet-cache", choices=['off', 'conservative', 'balanced', 'aggressive'],
                                default='off',
                                help="Default DeepCache profile. Reuses internal SDXL features between denoising steps.")

args_parser.parser.add_argument("--performance-log", type=str, nargs='?', const='performance.jsonl', default=None,
                                help="Write generation timing, peak VRAM and output hashes to a JSONL file.")
args_parser.parser.add_argument("--performance-run-label", type=str, default='default',
                                help="Label stored in performance benchmark records.")

args_parser.parser.add_argument("--tiled-upscale-batch-size", type=int, default=0,
                                help="Tiles processed together by Tiled SDXL Detail Upscale. 0 selects automatically.")

args_parser.parser.add_argument("--disable-latent-cache", action='store_true',
                                help="Disable the in-memory VAE latent cache.")
args_parser.parser.add_argument("--latent-cache-size", type=int, default=256, metavar='MB',
                                help="Maximum memory used by cached VAE latents in MB.")
args_parser.parser.add_argument("--disable-clip-cache", action='store_true',
                                help="Disable the in-memory CLIP condition cache.")
args_parser.parser.add_argument("--clip-cache-size", type=int, default=256, metavar='MB',
                                help="Maximum memory used by cached CLIP conditions in MB.")

args_parser.parser.add_argument("--vram-policy", choices=['auto', 'conservative', 'balanced', 'resident'],
                                default='auto', help="Control how aggressively models remain in VRAM.")

args_parser.parser.add_argument("--sam3-url", type=str, default='http://127.0.0.1:7866',
                                help="URL of the optional local SAM 3 mask worker.")
args_parser.parser.add_argument("--sam3-timeout", type=float, default=300.0,
                                help="Timeout in seconds for SAM 3 mask requests.")

args_parser.parser.add_argument("--always-download-new-model", action='store_true',
                                help="Always download newer models", default=False)

args_parser.parser.add_argument("--rebuild-hash-cache", help="Generates missing model and LoRA hashes.",
                                type=int, nargs="?", metavar="CPU_NUM_THREADS", const=-1)

args_parser.parser.set_defaults(
    disable_cuda_malloc=True,
    in_browser=True,
    port=None
)

args_parser.args = args_parser.parser.parse_args()

# (Disable by default because of issues like https://github.com/lllyasviel/Fooocus/issues/724)
args_parser.args.always_offload_from_vram = not args_parser.args.disable_offload_from_vram

if args_parser.args.disable_analytics:
    import os
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

if args_parser.args.disable_in_browser:
    args_parser.args.in_browser = False

args = args_parser.args
