"use client"

import { motion } from "framer-motion"
import type { VideoSettings } from "@/app/playground/page"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Slider } from "@/components/ui/slider"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface Step1BasicProps {
  settings: VideoSettings
  onSettingsChange: (key: keyof VideoSettings, value: any) => void
}

export function Step1Basic({ settings, onSettingsChange }: Step1BasicProps) {
  const genres = ["cyberpunk", "post-apocalyptic", "retro-punk", "steampunk", "dieselpunk"]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
      className="space-y-6"
    >
      <div>
        <Label htmlFor="subtitle-color" className="text-white mb-2 block">
          Subtitle Color
        </Label>
        <div className="flex items-center gap-4">
          <Input
            id="subtitle-color"
            type="color"
            value={settings.subtitleColor}
            onChange={(e) => onSettingsChange("subtitleColor", e.target.value)}
            className="w-16 h-10 p-1 bg-transparent border border-zinc-700 rounded-md"
          />
          <div className="h-10 flex-1 rounded-md" style={{ backgroundColor: settings.subtitleColor }}></div>
        </div>
      </div>

      <div>
        <div className="flex justify-between mb-2">
          <Label htmlFor="iterations" className="text-white">
            Number of Iterations
          </Label>
          <span className="text-pink-500 font-bold">{settings.iterations}</span>
        </div>
        <Slider
          id="iterations"
          min={1}
          max={10}
          step={1}
          value={[settings.iterations]}
          onValueChange={(value) => onSettingsChange("iterations", value[0])}
          className="py-4"
        />
        <div className="flex justify-between text-xs text-zinc-500">
          <span>Less detailed</span>
          <span>More detailed</span>
        </div>
      </div>

      <div>
        <Label htmlFor="genre" className="text-white mb-2 block">
          Genre
        </Label>
        <Select value={settings.genre} onValueChange={(value) => onSettingsChange("genre", value)}>
          <SelectTrigger className="bg-zinc-950 border-zinc-800 text-white">
            <SelectValue placeholder="Select a genre" />
          </SelectTrigger>
          <SelectContent className="bg-zinc-950 border-zinc-800 text-white">
            {genres.map((genre) => (
              <SelectItem key={genre} value={genre} className="capitalize">
                {genre}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </motion.div>
  )
}