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

local loc = {
	mod_title = {
		en = "Conductor (bridge)",
	},
	mod_description = {
		en = "Streams Darktide's music state to the Conductor app so it can score the game with your own playlist. Turn in-game music volume to 0.",
	},
	auto_start = {
		en = "Start Conductor with the game",
	},
	auto_start_desc = {
		en = "Launch the Conductor music app in the background when Darktide starts. It closes itself when the game closes.",
	},
	mute_game_music = {
		en = "Replace game music",
	},
	mute_game_music_desc = {
		en = "While Conductor is playing, silence Darktide's own music. Your music volume slider then controls Conductor instead. If Conductor stops, the game's music comes back.",
	},
	level = {
		en = "Conductor level",
	},
	level_desc = {
		en = "Fine-tune Conductor against the game. 100 = follow your Master and Music sliders exactly; raise above 100 to make your music louder, lower it to sit further back.",
	},
	duck_dialogue = {
		en = "Lower music under dialogue",
	},
	duck_dialogue_desc = {
		en = "Dip the music a little while the mission giver is talking (briefings, objective calls). Combat barks never duck the music.",
	},
	duck_voice_chat = {
		en = "Lower music under voice chat",
	},
	duck_voice_chat_desc = {
		en = "Dip the music while teammates are talking on voice chat (not when you talk).",
	},
	thumbs_up = {
		en = "Thumbs up current song",
	},
	thumbs_up_desc = {
		en = "Play this song more often. Press again to set it back to neutral.",
	},
	timing_up = {
		en = "Timing: good",
	},
	timing_up_desc = {
		en = "The last music change came at the right moment (or: staying put right now is right).",
	},
	timing_down = {
		en = "Timing: bad",
	},
	timing_down_desc = {
		en = "2+ changes in the last 15 s = too many. 1 change = it was the wrong moment / too late. No change = it should have changed by now.",
	},
	change_up = {
		en = "Transition: good",
	},
	change_up_desc = {
		en = "That transition worked (timing, song, how it joined). Saved as training data; the song rating is untouched.",
	},
	change_down = {
		en = "Transition: bad",
	},
	change_down_desc = {
		en = "That transition was bad (wrong moment, wrong song, ugly join). Saved as training data; the song keeps playing.",
	},
	thumbs_down = {
		en = "Thumbs down current song",
	},
	thumbs_down_desc = {
		en = "This song doesn't fit: skip it now and never play it again. Undo with `python -m conductor ratings --clear <title>`.",
	},
	echo_states = {
		en = "Show music changes in chat",
	},
	echo_states_desc = {
		en = "Print what the music is doing and why: the fight phase (calm, skirmish, horde, peak, Valkyrie, hub) and each song or part change.",
	},
	param_interval = {
		en = "Parameter update interval (s)",
	},
	param_interval_desc = {
		en = "How often continuous values (health, enemy count, attack intensity) are sent.",
	},
}

loc.playlist = { en = "Playlist" }
loc.playlist_desc = { en = "Which music folder Conductor plays from. To use your own songs, put them in a new folder inside mods/conductor/data/playlists - it shows up here (after a game restart) marked not processed; pick it and the app gets its songs ready. The Next playlist key switches straight away." }
loc.next_playlist = { en = "Next playlist" }
loc.add_playlist = { en = "Add playlist (Spotify link)" }
loc.add_playlist_desc = { en = "Paste a Spotify playlist link and press Enter. The Conductor app downloads it with spotDL, gets every song ready (a progress bar shows on screen) and switches to it when it's done. It appears in the playlist list after a game restart." }
loc.add_playlist_placeholder = { en = "https://open.spotify.com/playlist/..." }
loc.add_youtube = { en = "Add playlist (YouTube Music link)" }
loc.add_youtube_desc = { en = "Paste a YouTube Music (or YouTube) playlist link and press Enter. The Conductor app downloads it, gets every song ready (a progress bar shows on screen) and switches to it when it's done." }
loc.add_youtube_placeholder = { en = "https://music.youtube.com/playlist?list=..." }
loc.direction = { en = "Gameplay music direction" }
loc.direction_desc = { en = "Who decides when the music changes during missions. Conductor: anticipates fights, follows enemy numbers and changes songs as fights escalate. Native game: follows Darktide's own music states exactly, just with your music. Menus, hub, loadout and loading screens work the same either way." }
loc.direction_conductor = { en = "Conductor" }
loc.direction_native = { en = "Native game" }
loc.next_playlist_desc = { en = "Switch to the next playlist now. Chat confirms the name and how many songs it has." }
for i, pl in ipairs(read_playlists()) do
	local label = pl.name
	if pl.total > 0 and pl.ready == 0 then
		label = string.format("%s (%d songs, not processed - pick to process)", pl.name, pl.total)
	elseif pl.total > 0 and pl.ready < pl.total then
		label = string.format("%s (%d of %d ready)", pl.name, pl.ready, pl.total)
	elseif pl.total > 0 then
		label = string.format("%s (%d songs)", pl.name, pl.total)
	end
	loc["playlist_opt_" .. i] = { en = label }
end

return loc
