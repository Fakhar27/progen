"use client"

import { motion } from "framer-motion"
import type { VideoSettings } from "@/app/playground/page"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Volume2 } from "lucide-react"

interface Step3MusicProps {
  settings: VideoSettings
  onSettingsChange: (key: keyof VideoSettings, value: any) => void
}

export function Step3Music({ settings, onSettingsChange }: Step3MusicProps) {
  const musicOptions = [
    {
      value: "punk",
      label: "Punk Rock",
      description: "Fast-paced, aggressive guitar riffs and drums",
    },
    {
      value: "synthwave",
      label: "Synthwave",
      description: "Retro electronic music with 80s vibes",
    },
    {
      value: "industrial",
      label: "Industrial",
      description: "Harsh, mechanical sounds with distorted beats",
    },
    {
      value: "none",
      label: "No Music",
      description: "Generate video without background music",
    },
  ]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
    >
      <Label className="text-white mb-4 block">Background Music</Label>

      <RadioGroup
        value={settings.backgroundMusic}
        onValueChange={(value) => onSettingsChange("backgroundMusic", value)}
        className="space-y-3"
      >
        {musicOptions.map((option) => (
          <motion.div
            key={option.value}
            whileHover={{ scale: 1.02 }}
            className={`flex items-center space-x-3 rounded-lg border p-4 cursor-pointer transition-colors ${
              settings.backgroundMusic === option.value
                ? "border-pink-500 bg-zinc-800/50"
                : "border-zinc-800 bg-zinc-900"
            }`}
            onClick={() => onSettingsChange("backgroundMusic", option.value)}
          >
            <RadioGroupItem value={option.value} id={`music-${option.value}`} className="text-pink-500" />
            <div className="flex-1">
              <label
                htmlFor={`music-${option.value}`}
                className="text-white font-medium cursor-pointer flex items-center"
              >
                {option.label}
                {option.value !== "none" && <Volume2 className="ml-2 h-4 w-4 text-pink-500" />}
              </label>
              <p className="text-zinc-400 text-sm">{option.description}</p>
            </div>
          </motion.div>
        ))}
      </RadioGroup>
    </motion.div>
  )
}
