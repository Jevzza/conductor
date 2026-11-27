-- On-screen progress while the Conductor app downloads + processes a new playlist (2026-11-22).
local mod = get_mod("conductor")

local UIWorkspaceSettings = require("scripts/settings/ui/ui_workspace_settings")
local UIWidget = require("scripts/managers/ui/ui_widget")

local W, H = 420, 8

local HudElementConductorProgress = class("HudElementConductorProgress", "HudElementBase")

local scenegraph_definition = {
	screen = UIWorkspaceSettings.screen,
	conductor_progress = {
		parent = "screen",
		vertical_alignment = "top",
		horizontal_alignment = "center",
		size = { W, 40 },
		position = { 0, 90, 900 },
	},
}

local widget_definitions = {
	bar = UIWidget.create_definition({
		{
			pass_type = "text",
			value_id = "label",
			style_id = "label",
			value = "",
			style = {
				font_type = "proxima_nova_bold",
				font_size = 18,
				text_vertical_alignment = "top",
				text_horizontal_alignment = "center",
				size = { W, 22 },
				offset = { 0, 0, 2 },
				text_color = { 230, 226, 214, 190 },
			},
		},
		{
			pass_type = "rect",
			style_id = "back",
			style = {
				size = { W, H },
				offset = { 0, 26, 0 },
				color = { 160, 0, 0, 0 },
			},
		},
		{
			pass_type = "rect",
			style_id = "fill",
			style = {
				size = { 0, H },
				offset = { 0, 26, 1 },
				color = { 230, 186, 160, 90 },
			},
		},
	}, "conductor_progress"),
}

HudElementConductorProgress.init = function(self, parent, draw_layer, start_scale)
	HudElementConductorProgress.super.init(self, parent, draw_layer, start_scale, {
		scenegraph_definition = scenegraph_definition,
		widget_definitions = widget_definitions,
	})
end

HudElementConductorProgress.update = function(self, dt, t, ui_renderer, render_settings, input_service)
	HudElementConductorProgress.super.update(self, dt, t, ui_renderer, render_settings, input_service)
	local w = self._widgets_by_name.bar
	local a = mod.addlist
	local show = a ~= nil and (a.state == "running" or (mod.now_unix and mod.now_unix() or 0) < a.shown_until)
	w.visible = show and true or false
	if show then
		w.content.label = "Conductor - " .. (a.text or "")
		w.style.fill.size[1] = math.floor(W * math.max(0, math.min(1, a.frac or 0)))
		if a.state == "error" then
			w.style.fill.color = { 230, 200, 60, 50 }
		elseif a.state == "done" then
			w.style.fill.color = { 230, 120, 190, 100 }
		else
			w.style.fill.color = { 230, 186, 160, 90 }
		end
	end
	w.dirty = true
end

return HudElementConductorProgress
