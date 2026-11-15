/* Run with Node + Playwright. Exercises the shipped editor in a headless browser,
 * with Gradio's textbox event bridge simulated (no GPU/model imports required). */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const {chromium} = require('playwright');

(async () => {
    const project = path.resolve(__dirname, '..');
    const source = fs.readFileSync(path.join(project, 'modules/composition_ui.py'), 'utf8');
    const markup = source.match(/COMPOSITION_HTML = '''([\s\S]*?)'''/)[1];
    const surface = source.match(/COMPOSITION_CANVAS_HTML = '''([\s\S]*?)'''/)[1];
    const browser = await chromium.launch({headless: true,
        ...(process.env.COMPOSITION_CHROME ? {executablePath: process.env.COMPOSITION_CHROME} : {})});
    try {
        const page = await browser.newPage({viewport: {width: 1280, height: 1000}});
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        await page.setContent(`<style>
            :root {--body-text-color:#e8e8ed; --input-background-fill:#25252b; --border-color-primary:#555;
                --background-fill-secondary:#18181d; --background-fill-primary:#111116} body{background:#111116;color:#eee;font:14px Arial}
            main{box-sizing:border-box;display:grid;grid-template-columns:minmax(0,2fr) minmax(300px,1fr);gap:16px;margin:20px}
            aside{box-sizing:border-box;padding:16px;border:1px solid #444;border-radius:10px}
            aside details{border:1px solid #444;border-radius:8px;padding:12px;margin-bottom:12px}
            aside summary{font-weight:600;cursor:pointer} aside details[open] summary{margin-bottom:12px}
            aside .image-count{display:block;margin-top:16px} aside .image-count input{display:block;width:100%;margin-top:10px}
            #image_output_area{height:768px;min-width:0;border:1px solid #444;border-radius:8px}
            #final_gallery{height:100%;display:flex;justify-content:center;align-items:center}
            #final_gallery img{width:100%;height:100%;object-fit:contain}
            @media(max-width:600px){main{grid-template-columns:minmax(0,1fr);margin:4px}#image_output_area{height:500px}}
            ${fs.readFileSync(path.join(project, 'css/style.css'), 'utf8')}
            </style><main><section id="image_output_area"><div id="final_gallery" class="main_view"></div>
            <div id="composition_surface">${surface}</div></section>
            <aside><details id="aspect_ratios_accordion"><summary>Aspect Ratios · 1024 × 1024</summary>
            <label><input type="radio" name="ratio" checked> 1024 × 1024</label>
            <label><input type="radio" name="ratio"> 768 × 1344</label></details>
            <details id="composition_accordion" open><summary>Compose with boxes</summary>${markup}</details>
            <div id="composition_data" class="composition-data"><textarea>{"version":1,"enabled":false,"boxes":[]}</textarea></div>
            <label class="image-count">Image Number<input type="range" value="1" min="1" max="32"></label></aside></main>`);
        await page.addScriptTag({content: `window.gradioApp = () => document;
            window.onUiLoaded = callback => window.startComposition = callback;
            window.onAfterUiUpdate = callback => window.updateComposition = callback;`});
        await page.addScriptTag({path: path.join(project, 'javascript/composition.js')});
        await page.evaluate(() => {
            window.startComposition();
            document.querySelector('#composition_data textarea').addEventListener('input', event => {
                window.fooocusComposition.load(event.target.value);
            });
        });
        // Gradio 3 appends current output values to the JS arguments (currentTask here).
        const data = () => page.evaluate(() => JSON.parse(window.fooocusComposition.submit(['task', 'stale input', null], 2)[1]));
        const add = page.locator('#composition_editor [data-action="add"]');
        const canvasToggle = page.locator('#composition_editor [data-action="edit"]');
        const prompt = page.locator('.composition-box.selected textarea');
        await add.click();
        assert.equal(await page.locator('#composition_editor .composition-canvas').count(), 0);
        assert.equal(await page.locator('#image_output_area .composition-canvas').count(), 1);
        assert((await page.locator('.composition-canvas').boundingBox()).width > 500);
        await prompt.fill('a woman in a red jacket');
        assert.equal((await data()).boxes[0].prompt, 'a woman in a red jacket');
        assert.equal((await data()).enabled, true);
        const submitted = await page.evaluate(() => window.fooocusComposition.submit(
            ['session state', 'main prompt', '{"enabled":false}', 'old output task'], 3));
        assert.equal(submitted.length, 3);
        assert.equal(submitted[0], 'session state');
        assert.equal(submitted[1], 'main prompt');
        assert.equal(JSON.parse(submitted[2]).boxes[0].prompt, 'a woman in a red jacket');
        await add.click();
        await prompt.fill('a black dog, sitting');
        const beforeTyping = (await data()).boxes[1];
        await prompt.press('End');
        await prompt.press('Enter');
        await prompt.pressSequentially('looking left');
        await prompt.press('ArrowLeft');
        const afterTyping = (await data()).boxes[1];
        assert.equal(afterTyping.prompt, 'a black dog, sitting\nlooking left');
        assert.equal(afterTyping.x, beforeTyping.x);
        assert.equal(afterTyping.w, beforeTyping.w);
        assert.equal(await prompt.evaluate(node => document.activeElement === node), true);
        assert.equal(await page.locator('.composition-box').count(), 2);
        assert.equal(await page.locator('.composition-box.selected').evaluate(node =>
            getComputedStyle(node).borderTopColor), 'rgb(72, 167, 232)');
        assert.equal(await page.locator('.composition-box.selected').evaluate(node =>
            getComputedStyle(node).borderTopStyle), 'solid');

        // Move a box, then resize it via the visible corner handle.
        const beforeMove = (await data()).boxes[1];
        let rect = await page.locator('.composition-box.selected .composition-box-header').boundingBox();
        await page.mouse.move(rect.x + rect.width / 2, rect.y + 12);
        await page.mouse.down(); await page.mouse.move(rect.x + rect.width / 2 + 25, rect.y + 32, {steps: 5}); await page.mouse.up();
        const afterMove = (await data()).boxes[1];
        assert(afterMove.x > beforeMove.x && afterMove.y > beforeMove.y);
        const corner = await page.locator('.composition-box.selected .composition-handle.se').boundingBox();
        await page.mouse.move(corner.x + 6, corner.y + 6);
        await page.mouse.down(); await page.mouse.move(corner.x + 22, corner.y + 28, {steps: 5}); await page.mouse.up();
        const afterResize = (await data()).boxes[1];
        assert(afterResize.w > afterMove.w && afterResize.h > afterMove.h);

        // Coordinate edits and keyboard navigation remain bounded.
        await page.getByRole('spinbutton', {name: 'Box X percent'}).fill('100');
        await page.getByRole('spinbutton', {name: 'Box X percent'}).press('Tab');
        let box = (await data()).boxes[1];
        assert(box.x + box.w <= 1.000001);
        await page.locator('.composition-box.selected').focus();
        const oldX = box.x;
        await page.keyboard.press('ArrowLeft');
        assert((await data()).boxes[1].x < oldX);

        const saved = await data();
        await page.evaluate(() => window.fooocusComposition.dimensions('768×1344', -1, -1));
        assert.deepEqual(await data(), saved); // Resolution never changes normalized coordinates.
        rect = await page.locator('[data-role="canvas"]').boundingBox();
        assert(Math.abs(rect.width / rect.height - 768 / 1344) < .01);
        await page.evaluate(() => window.fooocusComposition.dimensions('768×1344', 1408, 704));
        rect = await page.locator('[data-role="canvas"]').boundingBox();
        assert(Math.abs(rect.width / rect.height - 2) < .01);

        // Browser text stays text, including strings that look like HTML.
        const hostile = '<img src=x onerror="window.boxInjected=true">\nFuchs, "rot"';
        await prompt.fill(hostile);
        assert.equal((await data()).boxes[1].prompt, hostile);
        assert.equal(await page.locator('#composition_editor img').count(), 0);
        assert.equal(await page.evaluate(() => window.boxInjected), undefined);

        // Metadata restore, including a queued older change event.
        await page.evaluate(value => {
            const bridge = document.querySelector('#composition_data textarea');
            bridge.value = JSON.stringify(value);
            window.fooocusComposition.load(bridge.value);
        }, saved);
        assert.deepEqual(await data(), saved);
        await prompt.fill('latest prompt before Generate');
        await page.evaluate(value => window.fooocusComposition.load(JSON.stringify(value)), saved);
        assert.equal((await data()).boxes[0].prompt, 'latest prompt before Generate');
        await page.locator('[data-role="enabled"]').uncheck();
        assert.equal((await data()).enabled, false);
        assert.equal((await data()).boxes.length, 2);

        for (let i = 0; i < 4; i++) await add.click();
        assert.equal((await data()).boxes.length, 6);
        assert.equal(await add.isDisabled(), true);
        await page.locator('[data-action="remove"]').click();
        assert.equal((await data()).boxes.length, 5);
        assert.equal(await add.isDisabled(), false);
        await page.locator('[data-action="clear"]').click();
        assert.equal((await data()).boxes.length, 0);
        assert.equal((await data()).enabled, false);

        // Drag on blank canvas creates a box.
        rect = await page.locator('[data-role="canvas"]').boundingBox();
        await page.mouse.move(rect.x + 15, rect.y + 15);
        await page.mouse.down(); await page.mouse.move(rect.x + rect.width / 2, rect.y + rect.height * .8, {steps: 8}); await page.mouse.up();
        assert.equal((await data()).boxes.length, 1);
        await prompt.fill('a woman in a red jacket');
        await add.click(); await prompt.fill('a black dog');
        await page.evaluate(() => window.fooocusComposition.dimensions('1024×1024', -1, -1));
        // During generation the existing preview/gallery stays usable. The layout survives.
        const beforeGeneration = await data();
        await page.evaluate(() => window.fooocusComposition.generating(true));
        assert.equal(await page.locator('#composition_workspace').isVisible(), false);
        assert.equal(await add.isDisabled(), true);
        for (const selector of ['[data-role="enabled"]', '[data-role="selection"]',
            '[data-role="strength"]', '[data-coordinate="x"]', '[data-action="remove"]',
            '[data-action="clear"]', '[data-action="edit"]']) {
            assert.equal(await page.locator(`#composition_editor ${selector}`).isDisabled(), true);
        }
        assert.equal(await page.locator('#final_gallery').isVisible(), true);
        await page.evaluate(() => window.fooocusComposition.generating(false));
        assert.equal(await page.locator('#composition_workspace').isVisible(), false);
        assert.equal(await page.locator('#composition_editor [data-role="strength"]').isDisabled(), false);
        assert.equal(await page.locator('#composition_editor [data-role="selection"]').isDisabled(), false);
        // Repeated toggles after completion must not re-enter the busy state.
        for (let i = 0; i < 3; i++) {
            await page.locator('[data-role="enabled"]').uncheck();
            assert.equal(await add.isDisabled(), false);
            await page.locator('[data-role="enabled"]').check();
            assert.equal(await page.locator('#composition_workspace').isVisible(), true);
            await page.locator('[data-action="view"]').click();
        }
        assert.equal(await canvasToggle.textContent(), 'Open canvas');
        assert.equal(await canvasToggle.getAttribute('aria-expanded'), 'false');
        assert.equal(await page.locator('#image_output_area button:visible').count(), 0);
        await canvasToggle.click();
        assert.equal(await canvasToggle.getAttribute('aria-expanded'), 'true');
        assert.deepEqual(await data(), beforeGeneration);
        await page.locator('[data-action="view"]').click();
        // A generated result stays in the gallery. The canvas shows the layout for the next image.
        await page.evaluate(async () => {
            const img = document.createElement('img');
            img.src = 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024"><rect width="1024" height="1024" fill="#304650"/></svg>');
            document.querySelector('#final_gallery').appendChild(img);
            await img.decode();
        });
        assert.equal(await page.locator('#final_gallery img').isVisible(), true);
        assert.equal(await page.locator('#image_output_area button:visible').count(), 0);
        await canvasToggle.click();
        assert.equal(await page.locator('.composition-canvas img').count(), 0);
        assert.deepEqual(await data(), beforeGeneration);
        const desktopScreenshot = process.env.COMPOSITION_SCREENSHOT || path.join(os.tmpdir(), 'fooocus-composition-editor.png');
        await page.screenshot({path: desktopScreenshot, fullPage: true});
        await page.setViewportSize({width: 390, height: 1100});
        await page.waitForFunction(() => document.querySelector('.composition-canvas').getBoundingClientRect().width <= 390);
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
        const narrow = await page.locator('.composition-canvas').boundingBox();
        assert(narrow.width <= 390);
        assert.equal(await page.locator('.composition-box.selected .composition-box-remove').isVisible(), true);
        // Delete an unselected box with the pointer, then delete the last box with Space.
        const beforeDeletion = await data();
        await page.locator('.composition-box:not(.selected)').getByRole('button', {name: 'Remove box 1'}).click();
        assert.deepEqual((await data()).boxes, [beforeDeletion.boxes[1]]);
        assert.equal(await prompt.inputValue(), beforeDeletion.boxes[1].prompt);
        const lastRemove = page.getByRole('button', {name: 'Remove box 1'});
        assert.equal(await lastRemove.evaluate(node => document.activeElement === node), true);
        await lastRemove.press('Space');
        assert.equal((await data()).boxes.length, 0);
        assert.equal((await data()).enabled, false);
        assert.equal(await page.locator('#composition_workspace').isVisible(), true);
        await page.locator('[data-action="view"]').click();
        assert.equal(await page.locator('#final_gallery img').isVisible(), true);
        assert.deepEqual(errors, []);
        console.log('Composition browser checks passed: inline typing, drag/resize, bounds, restore, generation visibility, settings canvas toggle, clean gallery, pointer and keyboard deletion, narrow layout.');
        console.log(`Screenshot: ${desktopScreenshot}`);
    } finally {
        await browser.close();
    }
})().catch(error => { console.error(error); process.exitCode = 1; });
