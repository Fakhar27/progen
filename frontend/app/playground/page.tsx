"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CustomizationPanel } from "../(main)/_components/customization-panel";
import { LoadingScreen } from "../(main)/_components/loading-screen";
import { PromptInput } from "../(main)/_components/prompt-input";
import { GlitchText } from "../(main)/_components/ui/glitch-text";
import SidebarWrapper from "../(main)/_components/SidebarWrapper"
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Menu, Download } from "lucide-react";
import Link from "next/link";

export type VideoSettings = {
  subtitleColor: string;
  iterations: number;
  genre: string;
  backgroundVideo: string;
  backgroundMusic: string;
  voiceType: string;
};

export function VideoGenerator() {
  const [prompt, setPrompt] = useState<string>("");
  const [settings, setSettings] = useState<VideoSettings>({
    subtitleColor: "#ff00ff",
    iterations: 2,
    genre: "futuristic",
    backgroundVideo: "1",
    backgroundMusic: "1",
    voiceType: "v2/en_speaker_6",
  });
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [videoData, setVideoData] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handlePromptSubmit = (value: string) => {
    setPrompt(value);
    setCurrentStep(1);
  };

  const handleSettingsChange = (key: keyof VideoSettings, value: any) => {
    setSettings((prev) => ({ ...prev, [key]: value }))
  }

  const generateVideo = async (prompt: string, settings: VideoSettings) => {
    try {
      setError(null); // Clear any previous errors
      
      const response = await fetch("/api/generate-video", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          prompt,
          genre: settings.genre,
          iterations: settings.iterations,
          backgroundType: settings.backgroundVideo,
          musicType: settings.backgroundMusic,
          voiceType: settings.voiceType,
          subtitleColor: settings.subtitleColor,
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to generate video');
      }

      return await response.json();
    } catch (error) {
      console.error('Error generating video:', error);
      throw error;
    }
  }

  const handleNextStep = async () => {
    if (currentStep < 4) {
      setCurrentStep(currentStep + 1);
    } else {
      try {
        // Show loading screen during generation
        setIsLoading(true);
        setVideoData(null);
        
        const result = await generateVideo(prompt, settings);
        console.log("Video generation successful");
        
        if (result.success && result.video_data) {
          // Store the video data
          setVideoData(result.video_data);
          // You can also save metrics if needed
          console.log("Video metrics:", result.metrics);
        } else {
          throw new Error("Video generation did not return valid data");
        }
      } catch (error) {
        console.error('Failed to generate video:', error);
        setError(error.message || "An unexpected error occurred");
      } finally {
        // Hide loading screen when done
        setIsLoading(false);
      }
    }
  }

  const handlePrevStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1)
    }
  }

  // Function to render the video player
  const renderVideoPlayer = () => {
    if (!videoData) return null;
    
    // Create a blob URL from the base64 data
    const blobUrl = `data:video/mp4;base64,${videoData}`;
    
    return (
      <div className="mt-8 p-6 bg-zinc-800 rounded-xl border border-zinc-700">
        <h3 className="text-xl font-bold text-white mb-4">Your Generated Video</h3>
        <video 
          className="w-full rounded-lg" 
          controls 
          autoPlay 
          src={blobUrl}
        />
        <div className="mt-4 flex justify-end">
          <a 
            href={blobUrl} 
            download="generated-video.mp4"
            className="px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 rounded-md text-white font-medium flex items-center gap-2"
          >
            <Download className="h-4 w-4" />
            Download Video
          </a>
        </div>
      </div>
    );
  };

  return (
    <div className="flex min-h-screen w-full">
      <div className="w-64">
        <SidebarWrapper />
      </div>
      <div className="flex-1 dark w-full min-h-screen bg-zinc-900">
        <div className="relative w-full max-w-7xl mx-auto dark:bg-zinc-900 p-6">
          
          <AnimatePresence>{isLoading && <LoadingScreen />}</AnimatePresence>

          {/* 🟣 Page Content */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="mb-8 text-center"
          >
            <GlitchText className="text-white text-5xl font-bold tracking-tighter">
              Generate Your AI Video 
            </GlitchText>
            <div className="h-1 w-48 bg-gradient-to-r from-purple-500 via-pink-500 to-red-500 mx-auto mt-2"></div>
            <p className="text-zinc-400 mt-4 max-w-2xl mx-auto">
              Enter your video idea below and customize it with our punk-style generator. Break the rules and create
              something revolutionary!
            </p>
          </motion.div>

          {/* Error message display */}
          {error && (
            <div className="mb-8 p-4 bg-red-900/50 border border-red-800 rounded-lg text-white">
              <p className="font-bold text-red-300">Error:</p>
              <p>{error}</p>
              <button 
                onClick={() => setError(null)}
                className="mt-2 px-4 py-2 bg-red-800 hover:bg-red-700 rounded-md text-white text-sm"
              >
                Dismiss
              </button>
            </div>
          )}

          {/* Main content area */}
          <div className="flex flex-col lg:flex-row gap-8">
            <motion.div
              className="flex-1"
              initial={{ opacity: 1 }}
              animate={{
                opacity: prompt ? 1 : 1,
                flex: prompt ? "0 0 40%" : "1 0 100%",
              }}
              transition={{ duration: 0.5 }}
            >
              <PromptInput value={prompt} onChange={setPrompt} onSubmit={handlePromptSubmit} />
            </motion.div>
            
            <AnimatePresence>
              {prompt && (
                <motion.div
                  className="flex-1 bg-zinc-900"
                  initial={{ opacity: 1, x: 50, width: 0 }}
                  animate={{ opacity: 1, x: 0, width: "auto" }}
                  exit={{ opacity: 0, x: 50, width: 0 }}
                  transition={{ duration: 0.5, type: "spring" }}
                >
                  <CustomizationPanel
                    settings={settings}
                    onSettingsChange={handleSettingsChange}
                    currentStep={currentStep}
                    onNextStep={handleNextStep}
                    onPrevStep={handlePrevStep}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Video player section */}
          {videoData && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              {renderVideoPlayer()}
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}

// ✅ Default export for Next.js
export default function PlaygroundPage() {
  return <VideoGenerator />;
}