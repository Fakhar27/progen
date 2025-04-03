"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import Image from "next/image"
import { MoreHorizontal, Play, Edit, Trash2, Clock } from "lucide-react"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"

interface RecentProjectsProps {
  className?: string
}

const projects = [
  {
    id: "1",
    title: "Cyberpunk City Tour",
    thumbnail: "/placeholder.svg?height=120&width=240",
    status: "completed",
    date: "2 days ago",
    duration: "0:45",
  },
  {
    id: "2",
    title: "Neon Wasteland",
    thumbnail: "/placeholder.svg?height=120&width=240",
    status: "completed",
    date: "1 week ago",
    duration: "1:20",
  },
  {
    id: "3",
    title: "Digital Rebellion",
    thumbnail: "/placeholder.svg?height=120&width=240",
    status: "processing",
    date: "Just now",
    duration: "0:30",
  },
  {
    id: "4",
    title: "Synthetic Dreams",
    thumbnail: "/placeholder.svg?height=120&width=240",
    status: "completed",
    date: "3 days ago",
    duration: "2:15",
  },
]

export function RecentProjects({ className = "" }: RecentProjectsProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className={`punk-card ${className}`}
    >
      <div className="p-6 border-b border-zinc-800">
        <h2 className="text-xl font-bold">Recent Projects</h2>
      </div>
      <div className="divide-y divide-zinc-800">
        {projects.map((project, index) => (
          <ProjectItem key={project.id} project={project} index={index} />
        ))}
      </div>
    </motion.div>
  )
}

interface ProjectItemProps {
  project: {
    id: string
    title: string
    thumbnail: string
    status: string
    date: string
    duration: string
  }
  index: number
}

function ProjectItem({ project, index }: ProjectItemProps) {
  const [isHovered, setIsHovered] = useState(false)

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.1 }}
      className="p-4 hover:bg-zinc-900/50 transition-colors"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="flex items-center gap-4">
        <div className="relative rounded-md overflow-hidden flex-shrink-0">
          <Image
            src={project.thumbnail || "/placeholder.svg"}
            alt={project.title}
            width={120}
            height={68}
            className="object-cover"
          />
          {project.status === "processing" ? (
            <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
              <div className="px-2 py-1 bg-yellow-500/20 rounded-md flex items-center">
                <Clock className="h-3 w-3 text-yellow-500 mr-1 animate-pulse" />
                <span className="text-yellow-500 text-xs">Processing</span>
              </div>
            </div>
          ) : (
            <div
              className={`absolute inset-0 bg-black/60 flex items-center justify-center opacity-0 transition-opacity duration-200 ${
                isHovered ? "opacity-100" : ""
              }`}
            >
              <button className="h-8 w-8 rounded-full bg-pink-500/80 flex items-center justify-center">
                <Play className="h-4 w-4 text-white" />
              </button>
            </div>
          )}
          <div className="absolute bottom-1 right-1 px-1.5 py-0.5 bg-black/70 rounded text-xs text-white">
            {project.duration}
          </div>
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="font-medium text-white truncate">{project.title}</h3>
          <p className="text-sm text-zinc-400 mt-1">{project.date}</p>
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button className="p-2 rounded-md hover:bg-zinc-800">
              <MoreHorizontal className="h-4 w-4 text-zinc-400" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="bg-zinc-950 border-zinc-800">
            <DropdownMenuItem className="flex items-center cursor-pointer hover:bg-zinc-900 text-white">
              <Edit className="mr-2 h-4 w-4 text-cyan-500" />
              <span>Edit</span>
            </DropdownMenuItem>
            <DropdownMenuItem className="flex items-center cursor-pointer hover:bg-zinc-900 text-white">
              <Trash2 className="mr-2 h-4 w-4 text-red-500" />
              <span>Delete</span>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </motion.div>
  )
}