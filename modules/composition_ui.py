"""Markup for the dependency-free composition editor; state lives in a Gradio textbox."""

COMPOSITION_HTML = '''
<div id="composition_editor" class="composition-editor">
  <label class="composition-toggle"><input type="checkbox" data-role="enabled"> Enable composition</label>
  <div class="composition-toolbar">
    <button type="button" data-action="add">+ Add Box</button>
    <button type="button" data-action="remove" disabled>Remove Box</button>
    <button type="button" data-action="clear" disabled>Clear</button>
    <button type="button" data-action="edit" aria-expanded="false" aria-controls="composition_workspace">Open canvas</button>
  </div>
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
    <div class="composition-canvas" data-role="canvas" role="group" aria-label="Composition canvas"></div>
  </div>
</div>
'''
