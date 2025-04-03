"use client"

import { motion } from "framer-motion"
import type { VideoSettings } from "@/app/playground/page"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Mic } from "lucide-react"

interface Step4VoiceProps {
  settings: VideoSettings
  onSettingsChange: (key: keyof VideoSettings, value: any) => void
}

export function Step4Voice({ settings, onSettingsChange }: Step4VoiceProps) {
  const voiceOptions = [
    {
      value: "male",
      label: "Male Voice",
      description: "Deep, gritty male voice narration",
    },
    {
      value: "female",
      label: "Female Voice",
      description: "Strong, confident female voice narration",
    },
  ]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
    >
      <Label className="text-white mb-4 block">Voice Type</Label>

      <RadioGroup
        value={settings.voiceType}
        onValueChange={(value: "male" | "female") => onSettingsChange("voiceType", value)}
        className="space-y-3"
      >
        {voiceOptions.map((option) => (
          <motion.div
            key={option.value}
            whileHover={{ scale: 1.02 }}
            className={`flex items-center space-x-3 rounded-lg border p-4 cursor-pointer transition-colors ${
              settings.voiceType === option.value ? "border-pink-500 bg-zinc-800/50" : "border-zinc-800 bg-zinc-900"
            }`}
            onClick={() => onSettingsChange("voiceType", option.value as "male" | "female")}
          >
            <RadioGroupItem value={option.value} id={`voice-${option.value}`} className="text-pink-500" />
            <div className="flex-1">
              <label
                htmlFor={`voice-${option.value}`}
                className="text-white font-medium cursor-pointer flex items-center"
              >
                {option.label}
                <Mic className="ml-2 h-4 w-4 text-pink-500" />
              </label>
              <p className="text-zinc-400 text-sm">{option.description}</p>
            </div>
          </motion.div>
        ))}
      </RadioGroup>
    </motion.div>
  )
}