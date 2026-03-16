import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Briefcase, Users, MessageSquare, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

const cards = [
  {
    title: "Find Jobs",
    subtitle: "Upload your CV and discover matching opportunities",
    icon: Briefcase,
    color: "from-blue-500/20 to-indigo-500/20 border-blue-500/30",
    iconColor: "text-blue-500",
    href: "/candidate",
  },
  {
    title: "Find Talent",
    subtitle: "Post jobs and discover qualified candidates",
    icon: Users,
    color: "from-emerald-500/20 to-teal-500/20 border-emerald-500/30",
    iconColor: "text-emerald-500",
    href: "/recruiter",
  },
  {
    title: "Chat Assistant",
    subtitle: "Ask about jobs, skills, and career guidance",
    icon: MessageSquare,
    color: "from-purple-500/20 to-pink-500/20 border-purple-500/30",
    iconColor: "text-purple-500",
    href: "/chat",
  },
] as const;

export function HomePage() {
  const nav = useNavigate();

  return (
    <div className="h-full flex items-center justify-center p-8">
      <div className="max-w-3xl w-full space-y-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center space-y-3"
        >
          <div className="flex items-center justify-center gap-2 text-primary">
            <Sparkles className="w-8 h-8" />
            <h1 className="text-4xl font-bold tracking-tight">JobNexus</h1>
          </div>
          <p className="text-muted-foreground text-lg">
            AI-Powered Job Recommendation Engine
          </p>
          <p className="text-sm text-muted-foreground max-w-md mx-auto">
            Semantic matching between CVs and job descriptions using knowledge graphs,
            cross-encoder reranking, and LLM-powered analysis.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {cards.map((card, i) => (
            <motion.button
              key={card.href}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 + i * 0.1 }}
              onClick={() => nav(card.href)}
              className={cn(
                "group relative p-6 rounded-xl border bg-gradient-to-br text-left",
                "hover:shadow-lg hover:scale-[1.02] transition-all duration-200",
                card.color
              )}
            >
              <card.icon className={cn("w-8 h-8 mb-3", card.iconColor)} />
              <h2 className="font-semibold text-lg">{card.title}</h2>
              <p className="text-sm text-muted-foreground mt-1">
                {card.subtitle}
              </p>
            </motion.button>
          ))}
        </div>
      </div>
    </div>
  );
}
