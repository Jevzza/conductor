--[[
	Conductor — Darktide bridge.

	This is the ONLY game-specific code in Conductor. It makes no musical decisions.
	It observes what Darktide tells its own music system (Wwise) and streams it, as
	JSON lines, to a file the Conductor core tails:

		%TEMP%\conductor_events.jsonl

	Event kinds (protocol v1):
		hello   {game, protocol}                       — bridge (re)started, file truncated
		state   {group, value}                         — a Wwise state group changed value
		stinger {event}                                — a one-shot music event fired
		param   {name, value}                          — a continuous music parameter (throttled)
		user    {action="rate", value=1|-1}            — player pressed thumbs up / down

	Where the data comes from:
		WwiseGameSyncManager:_set_state(group, state) is called every frame for every
		music state group (music_game_state, music_combat, music_zone, music_objective,
		event_intensity, minion_aggro_intensity, player_state, ...). We de-duplicate and
		only emit changes.
		WwiseWorld.set_global_parameter feeds player_health, player_attack_intensity,
		player_locked_in_melee, minion_counter_global.
		WwiseWorld.trigger_resource_event with a path under /music/ is a stinger.
]]

local mod = get_mod("conductor")

local PROTOCOL = 1
local MUSIC_PARAMS = {
	player_health          = true,
	player_attack_intensity = true,
	player_locked_in_melee = true,
	minion_counter_global  = true,
}

-- ─── Output file ─────────────────────────────────────────────────────────────

local _io = Mods and Mods.lua and Mods.lua.io
local _os = Mods and Mods.lua and Mods.lua.os
local temp_dir = (_os and (_os.getenv("TEMP") or _os.getenv("TMP"))) or "C:\\Windows\\Temp"
local OUT_PATH = temp_dir .. "\\conductor_events.jsonl"

local _file = nil
local _seq = 0

local function json_escape(s)
	return (tostring(s):gsub('[%c"\\]', function(c)
		if c == '"' then return '\\"' end
		if c == "\\" then return "\\\\" end
		return string.format("\\u%04x", string.byte(c))
	end))
end

local function json_value(v)
	local tv = type(v)
	if tv == "number" then
		if v ~= v or v == math.huge or v == -math.huge then return "null" end
		return string.format("%.4g", v)
	elseif tv == "boolean" then
		return v and "true" or "false"
	elseif v == nil then
		return "null"
	end
	return '"' .. json_escape(v) .. '"'
end

local function game_time()
	local ok, t = pcall(function() return Managers.time:time("main") end)
	return ok and t or nil
end

local function emit(kind, fields)
	if not _file then return end
	_seq = _seq + 1
	local parts = { '"k":' .. json_value(kind), '"seq":' .. _seq }
	local t = game_time()
	if t then parts[#parts + 1] = '"t":' .. json_value(t) end
	for key, value in pairs(fields) do
		parts[#parts + 1] = '"' .. key .. '":' .. json_value(value)
	end
	_file:write("{" .. table.concat(parts, ",") .. "}\n")
	_file:flush()
end

local function open_output()
	if not _io then
		mod:echo("[Conductor] Mods.lua.io unavailable — cannot reach the Conductor app.")
		return
	end
	if _file then pcall(function() _file:close() end) end
	-- Never lose a session: keep the previous log as a timestamped archive (the Conductor app
	-- moves archives into its recordings folder).
	pcall(function()
		local old = _io.open(OUT_PATH, "r")
		if old then
			local size = old:seek("end")
			old:close()
			if size and size > 0 and _os and _os.rename then
				local stamp = (_os.time and _os.time()) or 0
				_os.rename(OUT_PATH, temp_dir .. "\\conductor_events_" .. tostring(stamp) .. ".jsonl")
			end
		end
	end)
	_file = _io.open(OUT_PATH, "w")
	if not _file then
		mod:echo("[Conductor] Could not open " .. OUT_PATH)
		return
	end
	_seq = 0
	emit("hello", { game = "darktide", protocol = PROTOCOL })
end

-- ─── State tracking ──────────────────────────────────────────────────────────

local _last_state = {}     -- group -> value we last emitted
local _last_param = {}     -- name -> value we last emitted
local _pending_param = {}  -- name -> latest value not yet emitted
local _param_timer = 0

local function resend_snapshot()
	for group, value in pairs(_last_state) do
		emit("state", { group = group, value = value })
	end
end

mod:hook_safe("WwiseGameSyncManager", "_set_state", function(_self, group_name, new_state)
	if not mod:is_enabled() then return end
	if _last_state[group_name] == new_state then return end
	_last_state[group_name] = new_state
	emit("state", { group = group_name, value = new_state })
end)

mod:hook_safe("WwiseWorld", "set_global_parameter", function(_wwise_world, name, value)
	if MUSIC_PARAMS[name] then
		_pending_param[name] = value
	end
end)

mod:hook_safe("WwiseWorld", "trigger_resource_event", function(_wwise_world, event_name)
	if not mod:is_enabled() or type(event_name) ~= "string" then return end
	if event_name:find("/music/", 1, true) then
		emit("stinger", { event = event_name })
	end
end)

-- Never ask for the local player before the network is up: that crashes the ENGINE (pcall can't catch it).
local function _safe_local_player()
	local pm = Managers.player
	if not pm or not pm.local_player_safe then return nil end
	if not (Managers.state and Managers.state.extension) then return nil end   -- only once in a game world
	return pm:local_player_safe(1)
end

-- ─── Extra telemetry (dev data for tuning the music; every piece guarded) ─────────
-- enemies near me by type/distance, arrivals of specials/elites/monsters, my movement,
-- kills, toughness, ability use, team state, and mission metadata once per mission.
local _tel = { kills = {}, seen = {} }
local ENEMY_KEYS = {}
for _, kind in ipairs({ "horde", "other", "elite", "special", "monster" }) do
	for _, band in ipairs({ "8", "20", "40" }) do ENEMY_KEYS[#ENEMY_KEYS + 1] = "en_" .. kind .. "_" .. band end
end

local function _breed_of(unit)
	local ok, breed = pcall(function() return ScriptUnit.extension(unit, "unit_data_system"):breed() end)
	if not ok or not breed then return "other", nil end
	local tags = breed.tags or {}
	if tags.monster then return "monster", breed.name end
	if tags.special then return "special", breed.name end
	if tags.elite then return "elite", breed.name end
	if tags.horde then return "horde", breed.name end
	return "other", breed.name
end

local _tel_errs = {}
local function guard(name, fn)
	local ok, err = pcall(fn)
	if not ok and not _tel_errs[name] then
		_tel_errs[name] = true
		emit("tele_err", { where = name, err = tostring(err) })      -- once per block, to fix it
	end
end

local function _note_kill(now)
	_tel.kills[#_tel.kills + 1] = now
end

pcall(function()
	mod:hook_safe("AttackReportManager", "add_attack_result", function(_self, _dp, _attacked, attacking_unit, _dir, _pos, _weak, _dmg, attack_result)
		local me = _safe_local_player()
		if attack_result == "died" and me and attacking_unit == me.player_unit then _note_kill(game_time() or 0) end
	end)
	mod:hook_safe("AttackReportManager", "rpc_add_attack_result", function(_self, _ch, _dp, _a, _alu, _b, _dir, _pos, _weak, _dmg, attack_result_id)
		local res = NetworkLookup and NetworkLookup.attack_results and NetworkLookup.attack_results[attack_result_id]
		if res == "died" then _note_kill(game_time() or 0) end     -- reports sent to me are my own attacks
	end)
end)

local function telemetry_tick()
	local now = game_time()
	local player = Managers.player and _safe_local_player()
	local unit = player and player.player_unit
	if not now or not unit or not ALIVE[unit] then return end
	local pos = POSITION_LOOKUP[unit]
	if not pos then return end
	-- movement
	guard("movement", function()
		if _tel.last_pos and _tel.last_t and now > _tel.last_t then
			_pending_param.speed = Vector3.distance(pos, _tel.last_pos:unbox()) / (now - _tel.last_t)
		end
		_tel.last_pos, _tel.last_t = Vector3Box(pos), now
	end)
	-- enemies near me, by type and distance band (cumulative: within 8 / 20 / 40 m)
	guard("enemies", function()
		local side = Managers.state.extension:system("side_system").side_by_unit[unit]
		local list = side:relation_units("enemy") or {}          -- every enemy this client knows about
		local n = #list
		_tel.res = list
		local c = {}
		for _, k in ipairs(ENEMY_KEYS) do c[k] = 0 end
		local seen_now = {}
		for i = 1, n do
			local u = _tel.res[i]
			local upos = u and ALIVE[u] and POSITION_LOOKUP[u]
			if upos and Vector3.distance(pos, upos) <= 40 then
				local kind, name = _breed_of(u)
				local d = Vector3.distance(pos, upos)
				if d <= 8 then c["en_" .. kind .. "_8"] = c["en_" .. kind .. "_8"] + 1 end
				if d <= 20 then c["en_" .. kind .. "_20"] = c["en_" .. kind .. "_20"] + 1 end
				c["en_" .. kind .. "_40"] = c["en_" .. kind .. "_40"] + 1
				if kind == "special" or kind == "elite" or kind == "monster" then
					seen_now[u] = true
					if not _tel.seen[u] then
						emit("game", { event = "enemy_near", kind = kind, breed = name, dist = math.floor(d) })
					end
				end
			end
		end
		_tel.seen = seen_now
		for k, v in pairs(c) do _pending_param[k] = v end
	end)
	-- kills in the last 5 s
	guard("kills", function()
		local keep, k5 = {}, 0
		for _, t in ipairs(_tel.kills) do
			if now - t <= 5 then keep[#keep + 1] = t; k5 = k5 + 1 end
		end
		_tel.kills = keep
		_pending_param.kills_5s = k5
	end)
	-- toughness
	guard("toughness", function()
		local ext = ScriptUnit.has_extension(unit, "toughness_system")
		if ext then _pending_param.toughness = 100 * (ext:current_toughness_percent() or 0) end
	end)
	-- combat ability used (charges went down)
	guard("ability", function()
		local ext = ScriptUnit.has_extension(unit, "ability_system")
		if not ext then return end
		local ch = ext:remaining_ability_charges("combat_ability")
		if ch and _tel.charges and ch < _tel.charges then emit("game", { event = "ability" }) end
		_tel.charges = ch
	end)
	-- team: standing / down, humans / bots, nearest teammate
	guard("team", function()
		local standing, down, humans, bots, nearest = 0, 0, 0, 0, 999
		for _, p in pairs(Managers.player:players()) do
			local ok_h, human = pcall(function() return p:is_human_controlled() end)
			if ok_h and human then humans = humans + 1 else bots = bots + 1 end
			local u = p.player_unit
			if u and ALIVE[u] then
				local st = ScriptUnit.extension(u, "unit_data_system"):read_component("character_state").state_name
				if st == "knocked_down" or st == "hogtied" or st == "dead" or st == "netted" or st == "pounced"
					or st == "consumed" or st == "grabbed" or st == "mutant_charged" or st == "ledge_hanging" then
					down = down + 1
				else
					standing = standing + 1
				end
				if u ~= unit and POSITION_LOOKUP[u] then
					nearest = math.min(nearest, Vector3.distance(pos, POSITION_LOOKUP[u]))
				end
			else
				down = down + 1
			end
		end
		_pending_param.team_standing, _pending_param.team_down = standing, down
		_pending_param.humans, _pending_param.bots, _pending_param.nearest_ally = humans, bots, nearest
	end)
	-- mission metadata, once per mission
	guard("mission", function()
		local mm = Managers.state.mission
		local name = mm and mm:mission_name()
		if not name or name == _tel.mission then return end
		_tel.mission = name
		local info = { event = "mission", name = name }
		pcall(function() info.side_mission = mm:side_mission_name() end)
		pcall(function() info.challenge = Managers.state.difficulty:get_challenge() end)
		pcall(function() info.resistance = Managers.state.difficulty:get_resistance() end)
		pcall(function() info.havoc = Managers.state.difficulty:get_parsed_havoc_data() ~= nil end)
		pcall(function() info.circumstance = Managers.state.circumstance:circumstance_name() end)
		pcall(function() info.host = Managers.state.game_session:is_server() end)
		emit("game", info)
	end)
end

local _fps = { n = 0, t = 0, worst = 0 }
mod.update = function(dt)
	_fps.n, _fps.t, _fps.worst = _fps.n + 1, _fps.t + dt, math.max(_fps.worst, dt)
	if _fps.t >= 5 then                     -- frame rate, worst frame, and the most our telemetry cost (ms)
		_pending_param.fps = _fps.n / _fps.t
		_pending_param.worst_frame_ms = _fps.worst * 1000
		_pending_param.tele_ms = mod._tel_ms or 0
		_fps.n, _fps.t, _fps.worst, mod._tel_ms = 0, 0, 0, 0
	end
	if mod.check_dev_commands then mod.check_dev_commands(dt) end
	if mod.app_tick then mod.app_tick(dt) end
	if mod.duck_tick then mod.duck_tick(dt) end
	if mod.feedback_tick then mod.feedback_tick(dt) end
	if mod.addlist_tick then mod.addlist_tick(dt) end
	if mod.feed_tick then mod.feed_tick(dt) end
	_param_timer = _param_timer + dt
	local interval = mod:get("param_interval") or 0.5
	if _param_timer < interval then return end
	_param_timer = 0
	-- Proximity / pacing the game computes for its own music but never passes to Wwise:
	-- enemies aggroed near me, hordes near me, boss near me, the AI director's tension.
	pcall(function()
		local lp = _safe_local_player()
		local unit = lp and lp.player_unit
		local ext = unit and ScriptUnit.has_extension(unit, "music_parameter_system")
		if not ext then return end
		local b = function(x) return x and 1 or 0 end
		_pending_param.aggro_near = ext:num_aggroed_minions_near() or 0
		_pending_param.tension = ext:tension_percent() or 0
		_pending_param.horde_near = math.max(b(ext:ambush_horde_near()), b(ext:vector_horde_near()))
		_pending_param.boss_near = b(ext:boss_near())
	end)
	-- Anticipation: enemies physically near me, whether or not they've noticed me yet
	-- (-1 = the lookup isn't available on this client).
	local ok_near = pcall(function()
		local lp = _safe_local_player()
		local unit = lp and lp.player_unit
		if not unit then return end
		local bp = Managers.state.extension:system("broadphase_system").broadphase
		local side = Managers.state.extension:system("side_system").side_by_unit[unit]
		local names = side:relation_side_names("enemy")
		local pos = POSITION_LOOKUP[unit]
		mod._near_results = mod._near_results or {}
		table.clear(mod._near_results)
		_pending_param.enemies_8m = bp.query(bp, pos, 8, mod._near_results, names)
		table.clear(mod._near_results)
		_pending_param.enemies_20m = bp.query(bp, pos, 20, mod._near_results, names)
	end)
	if not ok_near then _pending_param.enemies_20m = -1 end
	local _c0 = _os and _os.clock and _os.clock()
	pcall(telemetry_tick)
	if _c0 then
		local ms = (_os.clock() - _c0) * 1000
		mod._tel_ms = math.max(mod._tel_ms or 0, ms)
	end
	for name, value in pairs(_pending_param) do
		local last = _last_param[name]
		if last == nil or math.abs(value - last) > 0.01 then
			_last_param[name] = value
			emit("param", { name = name, value = value })
		end
		_pending_param[name] = nil
	end
end

mod.on_enabled = function()
	resend_snapshot()
	if mod.on_setting_changed then mod.on_setting_changed() end
end

mod.on_unload = function()
	if _file then pcall(function() _file:close() end) end
	_file = nil
end

-- ─── Song ratings (thumbs up / down) ─────────────────────────────────────────
-- The bridge only forwards the press; the Conductor app owns the ratings. For chat feedback
-- we read the app's now-playing file: line 1 = title, line 2 = current rating (-1/0/1).

local STATUS_PATH = temp_dir .. "\\conductor_status.txt"

local function read_now_playing()
	if not _io then return nil end
	local f = _io.open(STATUS_PATH, "r")
	if not f then return nil end
	local title = f:read("*l")
	local rating = tonumber(f:read("*l") or "0") or 0
	f:close()
	if not title or title == "" then return nil end
	return title, rating
end

local function rate(value)
	if not mod:is_enabled() then return end
	local title, rating = read_now_playing()
	if not title then
		mod:echo("[Conductor] Nothing playing - is the Conductor app running?")
		return
	end
	emit("user", { action = "rate", value = value })
	-- The chat message comes from the app's confirmation (status line 4), so it always names
	-- the right song and the actual result. Watch for it for a few seconds.
	if mod.watch_feedback then mod.watch_feedback(3.0) end
end

-- ─── Confirmed feedback from the app → chat ──────────────────────────────────
local _feedback_seen = nil      -- last feedback id shown (nil = not yet baselined)
local _feedback_watch = 0

local function check_feedback()
	if not _io then return end
	local f = _io.open(STATUS_PATH, "r")
	if not f then return end
	f:read("*l"); f:read("*l"); f:read("*l")
	local line = f:read("*l")
	f:close()
	if not line then return end
	local id, text = line:match("^(%d+)|(.*)$")
	id = tonumber(id)
	if not id then return end
	if _feedback_seen == nil then
		_feedback_seen = id            -- first look after (re)load: don't repeat old messages
		return
	end
	if id ~= _feedback_seen then
		_feedback_seen = id
		if text and text ~= "" then
			mod:echo("[Conductor] " .. text)
		end
		_feedback_watch = 0
	end
end

mod.watch_feedback = function(seconds)
	if _feedback_seen == nil then check_feedback() end
	_feedback_watch = seconds
end

-- ─── Music feed → chat ("Show music changes in chat") ───────────────────────────
-- The app writes %TEMP%\conductor_feed.txt: "id|text" lines (phase changes, song changes and
-- why). Show the ones newer than what we've seen; on (re)load, start from the newest.
local FEED_PATH = temp_dir .. "\\conductor_feed.txt"
local _feed_seen = nil
local _feed_timer = 0

local function feed_tick(dt)
	_feed_timer = _feed_timer + dt
	if _feed_timer < 0.5 or not _io then return end
	_feed_timer = 0
	local f = _io.open(FEED_PATH, "r")
	if not f then return end
	local newest = _feed_seen or 0
	local show = mod:get("echo_states") and _feed_seen ~= nil
	for line in f:lines() do
		local id, text = line:match("^(%d+)|(.*)$")
		id = tonumber(id)
		if id and id > (_feed_seen or math.huge) and show then
			mod:echo("[Conductor] " .. text)
		end
		if id and id > newest then newest = id end
	end
	f:close()
	_feed_seen = newest
end

mod.feed_tick = feed_tick

mod.feedback_tick = function(dt)
	if _feedback_watch > 0 then
		_feedback_watch = _feedback_watch - dt
		check_feedback()
	end
end

mod.thumbs_up = function()
	rate(1)
end

mod.thumbs_down = function()
	rate(-1)
end

-- Rate the last music CHANGE (not the song): the app logs it for training
local function rate_change(value, action)
	if not mod:is_enabled() then return end
	if not read_now_playing() then
		mod:echo("[Conductor] Nothing playing - is the Conductor app running?")
		return
	end
	emit("user", { action = action or "rate_change", value = value })
	if mod.watch_feedback then mod.watch_feedback(3.0) end
end

mod.change_up = function()
	rate_change(1)
end

mod.change_down = function()
	rate_change(-1)
end

mod.timing_up = function()
	rate_change(1, "rate_timing")
end

mod.timing_down = function()
	rate_change(-1, "rate_timing")
end

local PLAYLISTS_TXT = [[..\mods\conductor\data\playlists.txt]]
local function read_playlist_names()
	local names = {}
	local f = _io and _io.open(PLAYLISTS_TXT, "r")
	if f then
		for line in f:lines() do
			local name = line:match("^(.-)|")
			if name and name ~= "" then names[#names + 1] = name end
		end
		f:close()
	end
	return names
end

-- The options dropdown is built from a table DMF keeps by reference and re-reads every time the
-- options view opens, so appending to it shows a newly added playlist without a game restart.
local function refresh_playlist_options()
	pcall(function()
		local dmf = get_mod("DMF")
		for _, mod_data in ipairs(dmf and dmf.options_widgets_data or {}) do
			if mod_data[1] and mod_data[1].mod_name == mod:get_name() then
				for i = 2, #mod_data do
					local w = mod_data[i]
					if w.setting_id == "playlist" and type(w.options) == "table" then
						local have = {}
						for _, o in ipairs(w.options) do have[o.value] = true end
						for _, n in ipairs(read_playlist_names()) do
							if not have[n] then w.options[#w.options + 1] = { text = n, value = n } end
						end
					end
				end
			end
		end
	end)
end
mod.refresh_playlist_options = refresh_playlist_options

-- Next playlist: cycle through the app's playlists.txt (read fresh, so new ones show up).
mod.next_playlist = function()
	if not mod:is_enabled() or not _io then return end
	local names = read_playlist_names()
	refresh_playlist_options()
	if #names < 2 then
		mod:echo("[Conductor] Only one playlist so far.")
		return
	end
	local cur, nxt = mod:get("playlist"), names[1]
	for i, n in ipairs(names) do
		if n == cur then nxt = names[i % #names + 1] end
	end
	mod:set("playlist", nxt, true)        -- on_setting_changed sends it; the app confirms in chat
	if mod.watch_feedback then mod.watch_feedback(4.0) end
end

-- ─── Dev command file (automated testing) ────────────────────────────────────
-- Write one command per line to %TEMP%\conductor_cmd.txt; checked once a second from
-- mod.update, so it works everywhere (hub, menus, missions), unlike HUD-driven watchers.
--   reload     reload all DMF mods (picks up code changes without restarting the game)
--   rate_up    same as pressing the thumbs-up keybind
--   rate_down  same as pressing the thumbs-down keybind
--   snapshot   re-send all current music states

local CMD_PATH = temp_dir .. "\\conductor_cmd.txt"
local _cmd_timer = 0

local function run_dev_command(cmd)
	if cmd == "reload" then
		local mm = Managers and Managers.mod
		if mm then
			emit("dev", { cmd = cmd, ok = true })
			mm._reload_requested = true
		else
			emit("dev", { cmd = cmd, ok = false, err = "no Managers.mod" })
		end
	elseif cmd == "rate_up" then
		rate(1)
		emit("dev", { cmd = cmd, ok = true })
	elseif cmd == "rate_down" then
		rate(-1)
		emit("dev", { cmd = cmd, ok = true })
	elseif cmd == "snapshot" then
		resend_snapshot()
		emit("dev", { cmd = cmd, ok = true })
	elseif cmd == "disable" or cmd == "enable" then
		-- Kill switch: back to vanilla (game music on, app not relaunched), or undo that.
		local on = (cmd == "enable")
		mod:set("mute_game_music", on, true)
		mod:set("auto_start", on, true)
		if mod.on_setting_changed then mod.on_setting_changed("mute_game_music") end
		mod:echo(on and "[Conductor] Re-enabled." or "[Conductor] Disabled - game music restored.")
		emit("dev", { cmd = cmd, ok = true })
	else
		emit("dev", { cmd = cmd, ok = false, err = "unknown command" })
	end
end

local function check_dev_commands(dt)
	_cmd_timer = _cmd_timer + dt
	if _cmd_timer < 1 or not _io then return end
	_cmd_timer = 0
	local f = _io.open(CMD_PATH, "r")
	if not f then return end
	local cmds = {}
	for line in f:lines() do
		line = line:gsub("%s+$", "")
		if line ~= "" then cmds[#cmds + 1] = line end
	end
	f:close()
	if #cmds == 0 then return end
	local w = _io.open(CMD_PATH, "w")
	if w then w:close() end
	for _, cmd in ipairs(cmds) do
		local ok, e = pcall(run_dev_command, cmd)
		if not ok then emit("dev", { cmd = cmd, ok = false, err = tostring(e) }) end
	end
end

-- ─── App lifecycle, volume and game-music replacement ────────────────────────
-- The Conductor app is started silently (no console window) via ShellExecute. It writes a
-- heartbeat into the status file; only while that heartbeat is fresh do we mute Darktide's
-- own music, so a crashed or closed app can never leave you in silence.

local HEARTBEAT_STALE = 8          -- seconds without a heartbeat = app not running

local _ffi = Mods and Mods.lua and Mods.lua.ffi
local _shell32 = nil
if _ffi then
	pcall(_ffi.cdef, [[
		void* ShellExecuteA(void* hwnd, const char* op, const char* file, const char* params,
		                    const char* dir, int show);
		unsigned long GetCurrentDirectoryA(unsigned long n, char* buf);
	]])
	local ok, lib = pcall(_ffi.load, "shell32")
	if ok then _shell32 = lib end
end

-- Drop-in install: everything lives in this mod's folder (mods/conductor). The game runs from
-- binaries/, so the mod folder is ..\mods\conductor; made absolute for ShellExecute.
local function mod_dir()
	local base = ".."
	if _ffi then
		pcall(function()
			local buf = _ffi.new("char[?]", 1024)
			local n = _ffi.C.GetCurrentDirectoryA(1024, buf)
			if n > 0 and n < 1024 then base = _ffi.string(buf, n) .. [[\..]] end
		end)
	end
	return base .. [[\mods\conductor]]
end
local MOD_DIR = mod_dir()
local APP_DIR = MOD_DIR .. [[\app]]
local APP_EXE = MOD_DIR .. [[\runtime\python\python.exe]]   -- console python, launched hidden: its ffmpeg/pip children share the hidden console
local APP_ARGS = "-m conductor.boot"

-- Linux (Proton): Conductor runs natively, started by the Steam launch option
-- (mods/conductor/linux/launch.sh). Never start the Windows python inside Wine.
local ON_PROTON = false
if _ffi then
	ON_PROTON = pcall(function()
		_ffi.cdef("const char* wine_get_version(void);")
		local v = _ffi.load("ntdll").wine_get_version()
		assert(v ~= nil)
	end)
end
if not ON_PROTON and _os and _os.getenv("STEAM_COMPAT_DATA_PATH") then ON_PROTON = true end

local function now_unix()
	local ok, t = pcall(function() return _os and _os.time() end)
	return ok and t or nil
end

local function app_alive()
	if not _io then return false end
	local f = _io.open(STATUS_PATH, "r")
	if not f then return false end
	f:read("*l")
	mod._app_idle = f:read("*l") == "idle"
	local beat = tonumber(f:read("*l") or "")
	f:close()
	local now = now_unix()
	if not beat or not now then return beat ~= nil end
	return now - beat <= HEARTBEAT_STALE
end

local function launch_app()
	if ON_PROTON then
		if not mod._proton_told then
			mod._proton_told = true
			mod:echo("[Conductor] Linux: start Conductor with the Steam launch option - see mods/conductor/README.txt")
		end
		return false
	end
	if not _shell32 then
		mod:echo("[Conductor] Can't start the app automatically (ffi unavailable).")
		return false
	end
	local ok, r = pcall(_shell32.ShellExecuteA, nil, "open", APP_EXE, APP_ARGS, APP_DIR, 0)
	emit("dev", { cmd = "launch", ok = ok and true or false })
	return ok
end

local function user_sound(key, default)
	local ok, v = pcall(function() return Application.user_setting("sound_settings", key) end)
	if ok and type(v) == "number" then return v end
	return default
end

local _alive = false
local _muted = false
local _music_param = nil       -- last music slider value the game pushed to Wwise
local _last_volume = nil
local _app_timer = 0
local _launched_at = nil

local function should_mute()
	return mod:is_enabled() and mod:get("mute_game_music") and _alive and not mod._app_idle
end

-- Darktide applies the music slider with Wwise.set_parameter("options_music_slider", v).
-- While Conductor plays we pass 0 instead, leaving the saved setting untouched.
mod:hook("Wwise", "set_parameter", function(func, name, value, ...)
	if name == "options_music_slider" then
		_music_param = value
		if should_mute() then
			return func(name, 0, ...)
		end
	end
	return func(name, value, ...)
end)

local function apply_mute()
	local want = should_mute()
	if want == _muted then return end
	_muted = want
	local v = _music_param or user_sound("options_music_slider", 100)
	pcall(Wwise.set_parameter, "options_music_slider", v)   -- goes through the hook above
end

local function send_volume(force)
	local v = {
		master  = user_sound("option_master_slider", 100),
		music   = user_sound("options_music_slider", 100),
		level   = mod:get("level") or 100,
		enabled = mod:is_enabled() and true or false,
		playlist = mod:get("playlist"),
		direction = mod:get("direction") or "conductor",
	}
	local key = string.format("%s|%s|%s|%s|%s|%s", v.master, v.music, v.level, tostring(v.enabled),
		tostring(v.playlist), tostring(v.direction))
	if force or key ~= _last_volume then
		_last_volume = key
		emit("volume", v)
	end
end

local function app_tick(dt)
	_app_timer = _app_timer + dt
	if _app_timer < 1 then return end
	_app_timer = 0
	_alive = app_alive()
	if not _alive and mod:is_enabled() and mod:get("auto_start") then
		local t = now_unix() or 0
		if not _launched_at or t - _launched_at > 60 then   -- don't spam launches
			_launched_at = t
			launch_app()
		end
	end
	send_volume(false)
	apply_mute()
end

mod.app_tick = app_tick

-- ─── Vanilla-style mixing: background mute + ducking ─────────────────────────
-- Background: the game's BackgroundMute sets the Wwise state options_mute_all = "true" when
-- the window loses focus (if "Mute in background" is on). We mirror exactly that decision.
local _bg_muted = false
mod:hook_safe("Wwise", "set_state", function(group, state)
	if group == "options_mute_all" then
		local muted = (state == "true")
		if muted ~= _bg_muted then
			_bg_muted = muted
			emit("focus", { muted = muted })
		end
	end
end)

-- Dialogue: only the mission giver (radio briefings / objective calls). Combat barks play
-- almost continuously in a fight, so ducking under them buried the music (65% of a mission).
local _vo_playing = false
mod:hook_safe("DialogueSystem", "update", function(self)
	local ok, playing = pcall(self.is_mission_giver_dialogue_playing, self)
	_vo_playing = ok and playing or false
end)

-- Voice chat: other players speaking (not your own mic).
local _speakers = {}
local _voice_listener = {}
function _voice_listener:on_participant_update(_channel, participant)
	if not participant or not participant.account_id then return end
	local me = nil
	pcall(function() local lp = Managers.player and Managers.player:local_player_safe(1); me = lp and lp:account_id() end)
	if participant.account_id == me then return end
	_speakers[participant.account_id] = participant.is_speaking and true or nil
end
pcall(function()
	Managers.event:register(_voice_listener, "chat_manager_participant_update", "on_participant_update")
end)

local _duck_sent = nil
local _duck_timer = 0
local function duck_tick(dt)
	_duck_timer = _duck_timer + dt
	if _duck_timer < 0.1 then return end
	_duck_timer = 0
	local voice = next(_speakers) ~= nil
	local vo = _vo_playing and mod:get("duck_dialogue") and true or false
	voice = voice and mod:get("duck_voice_chat") and true or false
	local key = tostring(vo) .. tostring(voice)
	if key ~= _duck_sent then
		_duck_sent = key
		emit("duck", { dialogue = vo, voice = voice })
	end
end
mod.duck_tick = duck_tick

local _orig_unload = mod.on_unload
mod.on_unload = function(...)
	pcall(function() Managers.event:unregister(_voice_listener, "chat_manager_participant_update") end)
	if _orig_unload then return _orig_unload(...) end
end

-- ─── Add a playlist: mod options text field → app downloads + processes it (2026-11-22) ─────────
local ADDLIST_PATH = temp_dir .. "\\conductor_addlist.txt"
mod.addlist = nil          -- { frac, text, state, name, shown_until } for the HUD bar
local _addlist_key, _addlist_timer = nil, 0

local function addlist_tick(dt)
	_addlist_timer = _addlist_timer + dt
	if _addlist_timer < 1 then return end
	_addlist_timer = 0
	if not _io then return end
	local f = _io.open(ADDLIST_PATH, "r")
	if not f then return end
	local line = f:read("*l")
	f:close()
	if not line or line == _addlist_key then return end
	local first = _addlist_key == nil
	_addlist_key = line
	local frac, text, state, name = line:match("^([%d%.]+)|(.-)|(%a+)|(.*)$")
	if not frac then return end
	if first and state ~= "running" then return end     -- an old finished run: don't replay it
	local t = now_unix() or 0
	mod.addlist = { frac = tonumber(frac) or 0, text = text, state = state, name = name,
		shown_until = state == "running" and math.huge or t + 10 }
	if state == "running" and not mod._job_running then
		mod:echo("[Conductor] " .. text)
	end
	mod._job_running = state == "running"
	if state == "done" and name ~= "" then
		mod:echo("[Conductor] " .. text .. " - switching to it")
		refresh_playlist_options()
		mod:set("playlist", name, false)
		send_volume(true)
		if mod.watch_feedback then mod.watch_feedback(6.0) end
	elseif state == "error" or state == "done" then
		mod:echo("[Conductor] " .. text)
	end
end
mod.addlist_tick = addlist_tick
mod.now_unix = now_unix

pcall(function()
	mod:register_hud_element({
		class_name = "HudElementConductorProgress",
		filename = "conductor/scripts/mods/conductor/HudElementConductorProgress",
		use_hud_scale = true,
		visibility_groups = { "alive", "dead", "tactical_overlay", "communication_wheel" },
	})
end)

mod.on_setting_changed = function(setting_id)
	if setting_id == "add_playlist" or setting_id == "add_youtube" then
		local url = mod:get(setting_id)
		if type(url) == "string" and url:match("%S") then
			emit("user", { action = "add_playlist", url = url })
			mod:echo("[Conductor] Sending the playlist to the app...")
			mod:set(setting_id, "", false)
			if mod.watch_feedback then mod.watch_feedback(5.0) end
		end
		return
	end
	if (setting_id == "playlist" or setting_id == "direction") and mod.watch_feedback then mod.watch_feedback(4.0) end
	send_volume(true)
	apply_mute()
end

mod.on_disabled = function()
	send_volume(true)
	apply_mute()
end

mod:command("conductor_start", "Start the Conductor music app", function()
	_launched_at = now_unix()
	if launch_app() then mod:echo("[Conductor] Starting the music app...") end
end)

mod.check_dev_commands = check_dev_commands

mod:command("conductor_snapshot", "Re-send current music state to the Conductor app", function()
	resend_snapshot()
	mod:echo("[Conductor] snapshot sent → " .. OUT_PATH)
end)

open_output()
