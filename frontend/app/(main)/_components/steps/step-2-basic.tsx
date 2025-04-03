"use client"

import { motion } from "framer-motion"
import type { VideoSettings } from "@/app/playground/page"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"

interface Step2BackgroundProps {
  settings: VideoSettings
  onSettingsChange: (key: keyof VideoSettings, value: any) => void
}

export function Step2Background({ settings, onSettingsChange }: Step2BackgroundProps) {
  const backgroundOptions = [
    {
      value: "urban",
      label: "Urban Dystopia",
      description: "Gritty city streets with neon signs and graffiti",
    },
    {
      value: "concert",
      label: "Concert Stage",
      description: "Energetic live performance setting with lights",
    },
    {
      value: "abstract",
      label: "Abstract Glitch",
      description: "Distorted digital patterns and visual noise",
    },
    {
      value: "industrial",
      label: "Industrial Wasteland",
      description: "Abandoned factories and rusty machinery",
    },
  ]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
    >
      <Label className="text-white mb-4 block">Background Video Type</Label>

      <RadioGroup
        value={settings.backgroundVideo}
        onValueChange={(value) => onSettingsChange("backgroundVideo", value)}
        className="space-y-3"
      >
        {backgroundOptions.map((option) => (
          <motion.div
            key={option.value}
            whileHover={{ scale: 1.02 }}
            className={`flex items-center space-x-3 rounded-lg border p-4 cursor-pointer transition-colors ${
              settings.backgroundVideo === option.value
                ? "border-pink-500 bg-zinc-800/50"
                : "border-zinc-800 bg-zinc-900"
            }`}
            onClick={() => onSettingsChange("backgroundVideo", option.value)}
          >
            <RadioGroupItem value={option.value} id={`background-${option.value}`} className="text-pink-500" />
            <div className="flex-1">
              <label htmlFor={`background-${option.value}`} className="text-white font-medium cursor-pointer">
                {option.label}
              </label>
              <p className="text-zinc-400 text-sm">{option.description}</p>
            </div>
          </motion.div>
        ))}
      </RadioGroup>
    </motion.div>
  )
}