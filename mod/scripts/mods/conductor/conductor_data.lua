local mod = get_mod("conductor")

-- Playlists come from the Conductor app (playlists.txt: "name|ready|total" per line).
local function read_playlists()
	local out = {}
	local io_ = Mods and Mods.lua and Mods.lua.io
	local f = io_ and io_.open([[..\mods\conductor\data\playlists.txt]], "r")
	if f then
		for line in f:lines() do
			local name, ready, total = line:match("^(.-)|(%d+)|(%d+)")
			if name and name ~= "" then
				out[#out + 1] = { name = name, ready = tonumber(ready), total = tonumber(total) }
			end
		end
		f:close()
	end
	if #out == 0 then out[1] = { name = "(none yet - add one below)", ready = 0, total = 0 } end
	-- DMF dropdowns need at least 2 choices (2026-11-24: one playlist = the whole options page failed);
	-- the app ignores a name it doesn't have
	if #out == 1 then out[2] = { name = "(add another playlist below)", ready = 0, total = 0 } end
	return out
end

local lists = read_playlists()
local playlist_options = {}
for i, pl in ipairs(lists) do
	playlist_options[i] = { text = "playlist_opt_" .. i, value = pl.name }
end

return {
	name         = mod:localize("mod_title"),
	description  = mod:localize("mod_description"),
	is_togglable = true,
	options = {
		widgets = {
			{
				setting_id    = "playlist",
				tooltip       = "playlist_desc",
				type          = "dropdown",
				default_value = lists[1].name,
				options       = playlist_options,
			},
			{
				setting_id       = "add_playlist",
				tooltip          = "add_playlist_desc",
				type             = "text",
				default_value    = "",
				placeholder_text = "add_playlist_placeholder",
				max_length       = 400,
			},
			{
				setting_id       = "add_youtube",
				tooltip          = "add_youtube_desc",
				type             = "text",
				default_value    = "",
				placeholder_text = "add_youtube_placeholder",
				max_length       = 400,
			},
			{
				setting_id    = "direction",
				tooltip       = "direction_desc",
				type          = "dropdown",
				default_value = "conductor",
				options       = {
					{ text = "direction_conductor", value = "conductor" },
					{ text = "direction_native",    value = "native" },
				},
			},
			{
				setting_id      = "next_playlist",
				tooltip         = "next_playlist_desc",
				type            = "keybind",
				keybind_trigger = "pressed",
				keybind_type    = "function_call",
				function_name   = "next_playlist",
				default_value   = {},
			},
			{
				setting_id    = "auto_start",
				tooltip       = "auto_start_desc",
				type          = "checkbox",
				default_value = true,
			},
			{
				setting_id    = "mute_game_music",
				tooltip       = "mute_game_music_desc",
				type          = "checkbox",
				default_value = true,
			},
			{
				setting_id      = "level",
				tooltip         = "level_desc",
				type            = "numeric",
				default_value   = 100,
				range           = { 0, 200 },
				decimals_number = 0,
			},
			{
				setting_id    = "duck_dialogue",
				tooltip       = "duck_dialogue_desc",
				type          = "checkbox",
				default_value = true,
			},
			{
				setting_id    = "duck_voice_chat",
				tooltip       = "duck_voice_chat_desc",
				type          = "checkbox",
				default_value = true,
			},
			{
				setting_id      = "thumbs_up",
				tooltip         = "thumbs_up_desc",
				type            = "keybind",
				keybind_trigger = "pressed",
				keybind_type    = "function_call",
				function_name   = "thumbs_up",
				default_value   = {},
			},
			{
				setting_id      = "thumbs_down",
				tooltip         = "thumbs_down_desc",
				type            = "keybind",
				keybind_trigger = "pressed",
				keybind_type    = "function_call",
				function_name   = "thumbs_down",
				default_value   = {},
			},
			{
				setting_id      = "change_up",
				tooltip         = "change_up_desc",
				type            = "keybind",
				keybind_trigger = "pressed",
				keybind_type    = "function_call",
				function_name   = "change_up",
				default_value   = {},
			},
			{
				setting_id      = "change_down",
				tooltip         = "change_down_desc",
				type            = "keybind",
				keybind_trigger = "pressed",
				keybind_type    = "function_call",
				function_name   = "change_down",
				default_value   = {},
			},
			{
				setting_id      = "timing_up",
				tooltip         = "timing_up_desc",
				type            = "keybind",
				keybind_trigger = "pressed",
				keybind_type    = "function_call",
				function_name   = "timing_up",
				default_value   = {},
			},
			{
				setting_id      = "timing_down",
				tooltip         = "timing_down_desc",
				type            = "keybind",
				keybind_trigger = "pressed",
				keybind_type    = "function_call",
				function_name   = "timing_down",
				default_value   = {},
			},
			{
				setting_id    = "echo_states",
				tooltip       = "echo_states_desc",
				type          = "checkbox",
				default_value = false,
			},
			{
				setting_id    = "param_interval",
				tooltip       = "param_interval_desc",
				type          = "numeric",
				default_value = 0.5,
				range         = { 0.1, 5 },
				decimals_number = 1,
				step_size_value = 0.1,
			},
		},
	},
}
