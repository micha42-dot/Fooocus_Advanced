"""Markup for the dependency-free composition editor; state lives in a Gradio textbox."""

COMPOSITION_HTML = '''
<div id="composition_editor" class="composition-editor">
  <label class="composition-toggle"><input type="checkbox" data-role="enabled"> Enable composition</label>
  <p class="composition-help">Place boxes and type their prompts directly on the large image area. Use the main prompt for the shared scene, light and style.</p>
  <div class="composition-toolbar">
    <button type="button" data-action="add">+ Add Box</button>
    <button type="button" data-action="remove" disabled>Remove Box</button>
    <button type="button" data-action="clear" disabled>Clear</button>
    <button type="button" data-action="edit">Edit boxes on image</button>
  </div>
  <div class="composition-status" data-role="status" aria-live="polite">No boxes</div>
  <label class="composition-field">Selected box <select data-role="selection" aria-label="Selected box"></select></label>
  <fieldset data-role="fields" disabled>
    <div class="composition-coordinates">
      <label>X %<input data-coordinate="x" type="number" min="0" max="98" step="1" aria-label="Box X percent"></label>
      <label>Y %<input data-coordinate="y" type="number" min="0" max="98" step="1" aria-label="Box Y percent"></label>
      <label>Width %<input data-coordinate="w" type="number" min="2" max="100" step="1" aria-label="Box width percent"></label>
      <label>Height %<input data-coordinate="h" type="number" min="2" max="100" step="1" aria-label="Box height percent"></label>
    </div>
    <label class="composition-field">Box strength <output data-role="strength-value">1.0</output>
      <input data-role="strength" type="range" min="0.1" max="3" step="0.1" value="1" aria-label="Box strength">
    </label>
  </fieldset>
  <p class="composition-help">Drag a box header to move it; drag corners to resize. Write the prompt inside the box. Up to 6 boxes. Empty prompts are ignored.</p>
  <p class="composition-help">Local text-to-image and Image Prompt. Regions guide placement, not exact object outlines. More boxes take longer. Disable for API generation or image editing.</p>
</div>
'''

COMPOSITION_CANVAS_HTML = '''
<div id="composition_workspace" class="composition-workspace" hidden>
  <div class="composition-workspace-toolbar">
    <span>Compose with boxes</span>
    <button type="button" data-action="add">+ Add Box</button>
    <button type="button" data-action="view">Show image</button>
  </div>
  <div class="composition-stage">
    <div class="composition-canvas" data-role="canvas" role="group" aria-label="Composition canvas">
      <img class="composition-reference" alt="Current generated image" hidden>
      <span class="composition-empty">Add a box or drag to draw one. Type prompts inside the boxes.</span>
    </div>
  </div>
</div>
<button id="composition_edit_image" class="composition-edit-image" type="button" hidden>Edit boxes on image</button>
'''
