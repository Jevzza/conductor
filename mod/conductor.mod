return {
	run = function()
		fassert(rawget(_G, "new_mod"), "`conductor` encountered an error loading the Darktide Mod Framework.")

		new_mod("conductor", {
			mod_script       = "conductor/scripts/mods/conductor/conductor",
			mod_data         = "conductor/scripts/mods/conductor/conductor_data",
			mod_localization = "conductor/scripts/mods/conductor/conductor_localization",
		})
	end,
	packages = {},
}
