/* Box composition editor. No canvas/framework dependency; prompts are always text. */
(function () {
    'use strict';
    const MAX_BOXES = 6;
    const MIN_SIZE = 0.02;
    const colors = ['#e88043', '#48a7e8', '#9b82ef', '#3dbb95', '#d665aa', '#b0a337'];
    const empty = () => ({version: 1, enabled: false, boxes: []});
    const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
    let state = empty();
    let selected = -1;
    let root = null;
    let canvas = null;
    let bridge = null;
    let lastSent = '';
    let ratio = 1;
    let resolution = '1024 × 1024';
    let drag = null;
    let workspace = null;
    let stage = null;
    let outputArea = null;
    let editButton = null;
    let editing = false;
    let busy = false;
    let resizeObserver = null;

    function constrain(box) {
        box.w = clamp(box.w, MIN_SIZE, 1);
        box.h = clamp(box.h, MIN_SIZE, 1);
        box.x = clamp(box.x, 0, 1 - box.w);
        box.y = clamp(box.y, 0, 1 - box.h);
        return box;
    }

    function read(value) {
        try {
            const parsed = typeof value === 'string' ? JSON.parse(value) : value;
            if (!parsed || parsed.version !== 1 || !Array.isArray(parsed.boxes) || parsed.boxes.length > MAX_BOXES) return empty();
            const boxes = parsed.boxes.map(box => {
                if (!['x', 'y', 'w', 'h', 'strength'].every(k => typeof box[k] === 'number' && Number.isFinite(box[k]))) throw Error('Invalid box');
                return constrain({x: box.x, y: box.y, w: box.w, h: box.h,
                    strength: clamp(box.strength, 0.1, 3), prompt: String(box.prompt || '').slice(0, 2000)});
            });
            return {version: 1, enabled: parsed.enabled === true, boxes};
        } catch (_) { return empty(); }
    }

    function publish() {
        lastSent = JSON.stringify(state);
        if (bridge && bridge.value !== lastSent) {
            // Gradio 3 listens to the native textarea's input event.
            Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set.call(bridge, lastSent);
            bridge.dispatchEvent(new Event('input', {bubbles: true}));
        }
    }

    const role = name => root.querySelector(`[data-role="${name}"]`);
    const current = () => state.boxes[selected];

    function fitCanvas() {
        if (!stage || workspace.hidden) return;
        const bounds = stage.getBoundingClientRect();
        const width = Math.max(1, Math.min(bounds.width, bounds.height * ratio));
        canvas.style.width = `${width}px`;
        canvas.style.height = `${width / ratio}px`;
    }

    function showWorkspace() {
        const visible = editing && !busy;
        workspace.hidden = !visible;
        outputArea.classList.toggle('composition-editing', visible);
        editButton.hidden = visible || busy || !state.enabled;
        root.querySelector('[data-action="edit"]').textContent = visible ? 'Show image' : 'Edit boxes on image';
        fitCanvas();
    }

    function openEditor() {
        if (busy) return;
        // Use the visible gallery image as a reference. Never alter the saved image.
        const images = [...outputArea.querySelectorAll('#final_gallery img')]
            .filter(img => img.naturalWidth && img.getBoundingClientRect().width > 100
                && img.getBoundingClientRect().height > 100 && !img.closest('.thumbnails'));
        images.sort((a, b) => b.getBoundingClientRect().width * b.getBoundingClientRect().height
            - a.getBoundingClientRect().width * a.getBoundingClientRect().height);
        const reference = canvas.querySelector('.composition-reference');
        const image = images[0];
        if (image && Math.abs(image.naturalWidth / image.naturalHeight / ratio - 1) < .02) {
            reference.src = image.currentSrc || image.src;
            reference.hidden = false;
        } else {
            reference.hidden = true;
            reference.removeAttribute('src');
        }
        editing = true;
        showWorkspace();
    }

    function focusPrompt() {
        const input = canvas.querySelector(`[data-index="${selected}"] textarea`);
        if (input) input.focus({preventScroll: true});
    }

    function renderCanvas() {
        canvas.querySelectorAll('.composition-box').forEach(node => {
            if (Number(node.dataset.index) >= state.boxes.length) node.remove();
        });
        canvas.style.setProperty('--composition-ratio', String(ratio));
        canvas.classList.toggle('composition-disabled', !state.enabled);
        canvas.querySelector('.composition-empty').hidden = state.boxes.length > 0;
        state.boxes.forEach((box, i) => {
            let node = canvas.querySelector(`[data-index="${i}"]`);
            if (!node) {
                node = document.createElement('div');
                node.dataset.index = String(i);
                node.tabIndex = 0;
                node.setAttribute('role', 'group');
                const header = document.createElement('div');
                header.className = 'composition-box-header';
                header.textContent = `Box ${i + 1} · drag to move`;
                node.appendChild(header);
                const prompt = document.createElement('textarea');
                prompt.className = 'composition-box-prompt';
                prompt.placeholder = 'Write the prompt for this box…';
                prompt.maxLength = 2000;
                prompt.setAttribute('aria-label', `Box ${i + 1} prompt`);
                prompt.addEventListener('focus', () => {
                    selected = Number(node.dataset.index); renderCanvas(); renderFields();
                });
                prompt.addEventListener('input', () => {
                    state.boxes[Number(node.dataset.index)].prompt = prompt.value;
                    renderFields(); publish();
                });
                node.appendChild(prompt);
                for (const corner of ['nw', 'ne', 'sw', 'se']) {
                    const handle = document.createElement('span');
                    handle.className = `composition-handle ${corner}`;
                    handle.dataset.corner = corner;
                    node.appendChild(handle);
                }
                canvas.appendChild(node);
            }
            node.className = 'composition-box' + (i === selected ? ' selected' : '');
            node.setAttribute('aria-label', `Box ${i + 1}`);
            Object.assign(node.style, {left: `${box.x * 100}%`, top: `${box.y * 100}%`,
                width: `${box.w * 100}%`, height: `${box.h * 100}%`});
            node.style.setProperty('--box-color', colors[i]);
            const prompt = node.querySelector('textarea');
            if (prompt.value !== box.prompt) prompt.value = box.prompt;
        });
        role('status').textContent = `${state.boxes.length} / ${MAX_BOXES} boxes · ${resolution}${state.enabled ? '' : ' · disabled'}`;
    }

    function renderFields() {
        const box = current();
        role('enabled').checked = state.enabled;
        role('enabled').disabled = busy;
        for (const host of [root, workspace]) {
            host.querySelector('[data-action="add"]').disabled = busy || state.boxes.length >= MAX_BOXES;
        }
        root.querySelector('[data-action="remove"]').disabled = busy || !box;
        root.querySelector('[data-action="clear"]').disabled = busy || !state.boxes.length;
        root.querySelector('[data-action="edit"]').disabled = busy;
        const select = role('selection');
        select.replaceChildren();
        state.boxes.forEach((item, i) => {
            const option = document.createElement('option');
            option.value = String(i);
            option.textContent = `Box ${i + 1}${item.prompt ? ': ' + item.prompt.slice(0, 40) : ''}`;
            select.appendChild(option);
        });
        select.value = String(selected);
        select.disabled = busy || !state.boxes.length;
        role('fields').disabled = busy || !box;
        role('strength').value = box ? box.strength : 1;
        role('strength-value').textContent = Number(box ? box.strength : 1).toFixed(1);
        root.querySelectorAll('[data-coordinate]').forEach(input => {
            input.value = box ? Math.round(box[input.dataset.coordinate] * 1000) / 10 : '';
        });
    }

    function render() { renderCanvas(); renderFields(); showWorkspace(); }

    function add(box) {
        if (busy || state.boxes.length >= MAX_BOXES) return;
        if (!editing) openEditor();
        const offset = (state.boxes.length % 4) * 0.08;
        state.boxes.push(box || {x: 0.12 + offset, y: 0.12 + offset, w: 0.4, h: 0.5, strength: 1, prompt: ''});
        selected = state.boxes.length - 1;
        state.enabled = true;
        render(); publish();
    }

    function point(event) {
        const bounds = canvas.getBoundingClientRect();
        return {x: clamp((event.clientX - bounds.left) / bounds.width, 0, 1),
            y: clamp((event.clientY - bounds.top) / bounds.height, 0, 1)};
    }

    function init() {
        const app = typeof gradioApp === 'function' ? gradioApp() : document;
        const found = app.querySelector('#composition_editor');
        const textbox = app.querySelector('#composition_data textarea');
        const surface = app.querySelector('#composition_workspace');
        if (!found || !textbox || !surface || (root === found && found.dataset.initialized)) return;
        root = found; bridge = textbox; workspace = surface;
        canvas = workspace.querySelector('[data-role="canvas"]');
        stage = workspace.querySelector('.composition-stage');
        outputArea = app.querySelector('#image_output_area');
        editButton = app.querySelector('#composition_edit_image');
        root.dataset.initialized = 'true';
        state = read(bridge.value); lastSent = bridge.value;
        selected = state.boxes.length ? 0 : -1;
        role('enabled').addEventListener('change', event => {
            state.enabled = event.target.checked;
            if (state.enabled) openEditor(); else editing = false;
            render(); publish();
        });
        for (const host of [root, workspace]) {
            host.querySelector('[data-action="add"]').addEventListener('click', () => {
                add(); focusPrompt(); outputArea.scrollIntoView({block: 'nearest', behavior: 'smooth'});
            });
        }
        root.querySelector('[data-action="edit"]').addEventListener('click', () => {
            if (editing) { editing = false; showWorkspace(); } else openEditor();
        });
        workspace.querySelector('[data-action="view"]').addEventListener('click', () => { editing = false; showWorkspace(); });
        editButton.addEventListener('click', openEditor);
        const reset = app.querySelector('#reset_button');
        if (reset) reset.addEventListener('click', () => window.fooocusComposition.generating(false));
        root.querySelector('[data-action="remove"]').addEventListener('click', () => {
            if (!current()) return;
            state.boxes.splice(selected, 1); selected = Math.min(selected, state.boxes.length - 1);
            if (!state.boxes.length) state.enabled = false;
            render(); publish();
        });
        root.querySelector('[data-action="clear"]').addEventListener('click', () => {
            state = empty(); selected = -1; render(); publish();
        });
        role('selection').addEventListener('change', event => {
            selected = Number(event.target.value); if (!editing) openEditor(); render(); focusPrompt();
        });
        role('strength').addEventListener('input', event => {
            if (!current()) return;
            current().strength = Number(event.target.value);
            role('strength-value').textContent = current().strength.toFixed(1); publish();
        });
        root.querySelectorAll('[data-coordinate]').forEach(input => input.addEventListener('change', () => {
            if (!current() || !Number.isFinite(input.valueAsNumber)) { renderFields(); return; }
            const key = input.dataset.coordinate;
            current()[key] = input.valueAsNumber / 100;
            constrain(current()); render(); publish();
        }));
        canvas.addEventListener('pointerdown', event => {
            if (busy || event.button !== 0 || event.target.closest('textarea')) return;
            const node = event.target.closest('.composition-box');
            const start = point(event);
            if (node) {
                selected = Number(node.dataset.index);
                drag = {start, box: {...current()}, corner: event.target.dataset.corner, drawing: false};
                render();
            } else {
                if (state.boxes.length >= MAX_BOXES) return;
                drag = {start, drawing: true, created: false};
            }
            canvas.setPointerCapture(event.pointerId);
            event.preventDefault();
        });
        canvas.addEventListener('pointermove', event => {
            if (!drag) return;
            const p = point(event);
            const dx = p.x - drag.start.x, dy = p.y - drag.start.y;
            if (drag.drawing) {
                if (!drag.created) {
                    if (Math.abs(dx) < MIN_SIZE || Math.abs(dy) < MIN_SIZE) return;
                    add({x: 0, y: 0, w: MIN_SIZE, h: MIN_SIZE, strength: 1, prompt: ''});
                    drag.created = true;
                }
                Object.assign(current(), {x: Math.min(p.x, drag.start.x), y: Math.min(p.y, drag.start.y),
                    w: Math.max(MIN_SIZE, Math.abs(dx)), h: Math.max(MIN_SIZE, Math.abs(dy))});
            } else {
                const b = drag.box;
                const c = drag.corner;
                if (!c) { current().x = b.x + dx; current().y = b.y + dy; }
                else {
                    const x0 = c.includes('w') ? clamp(b.x + dx, 0, b.x + b.w - MIN_SIZE) : b.x;
                    const y0 = c.includes('n') ? clamp(b.y + dy, 0, b.y + b.h - MIN_SIZE) : b.y;
                    const x1 = c.includes('e') ? clamp(b.x + b.w + dx, b.x + MIN_SIZE, 1) : b.x + b.w;
                    const y1 = c.includes('s') ? clamp(b.y + b.h + dy, b.y + MIN_SIZE, 1) : b.y + b.h;
                    Object.assign(current(), {x: x0, y: y0, w: x1 - x0, h: y1 - y0});
                }
            }
            constrain(current()); render();
        });
        function finishDrag(event) {
            if (!drag) return;
            drag = null;
            if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
            publish();
        }
        canvas.addEventListener('pointerup', finishDrag);
        canvas.addEventListener('pointercancel', finishDrag);
        canvas.addEventListener('keydown', event => {
            if (event.target.closest('textarea')) return;
            const node = event.target.closest('.composition-box');
            if (!node) return;
            selected = Number(node.dataset.index);
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault(); render(); focusPrompt(); return;
            }
            if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
            event.preventDefault();
            const horizontal = event.key === 'ArrowLeft' || event.key === 'ArrowRight';
            const sign = event.key === 'ArrowLeft' || event.key === 'ArrowUp' ? -1 : 1;
            const key = event.shiftKey ? (horizontal ? 'w' : 'h') : (horizontal ? 'x' : 'y');
            current()[key] += sign * 0.01;
            constrain(current()); render(); publish();
            canvas.querySelector(`[data-index="${selected}"]`).focus();
        });
        if (resizeObserver) resizeObserver.disconnect();
        resizeObserver = new ResizeObserver(fitCanvas);
        resizeObserver.observe(stage);
        editing = state.enabled;
        render();
    }

    window.fooocusComposition = {
        load(value, force = false) {
            init();
            if (!root || value === lastSent) return;
            // Ignore a queued change callback older than the current textbox value.
            if (!force && bridge.value !== value) return;
            state = read(value); lastSent = value;
            selected = state.boxes.length ? 0 : -1;
            editing = state.enabled && !busy;
            render();
        },
        dimensions(value, width, height) {
            const numbers = String(value).match(/\d+/g) || [1024, 1024];
            const w = Number(width) > 0 ? Number(width) : Number(numbers[0]);
            const h = Number(height) > 0 ? Number(height) : Number(numbers[1]);
            if (w > 0 && h > 0) { ratio = w / h; resolution = `${w} × ${h}`; }
            init(); if (canvas) {
                canvas.querySelector('.composition-reference').hidden = true;
                renderCanvas(); fitCanvas();
            }
        },
        generating(value) {
            init(); busy = Boolean(value);
            if (busy) { editing = false; drag = null; }
            if (root) render();
        },
        submit(args, inputCount) {
            init();
            // Gradio 3 supplies inputs AND current output values to _js callbacks.
            // Only inputs may be returned to Python; composition is the last INPUT.
            const inputs = args.slice(0, inputCount);
            if (root) { publish(); inputs[inputs.length - 1] = JSON.stringify(state); }
            return inputs;
        }
    };
    if (typeof onUiLoaded === 'function') onUiLoaded(init);
    if (typeof onAfterUiUpdate === 'function') onAfterUiUpdate(init);
})();
