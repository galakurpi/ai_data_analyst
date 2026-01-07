import * as React from "react"
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from "@/components/ui/input-group"
import { HugeiconsIcon } from "@hugeicons/react"
import { ArrowRight01Icon, ChartIcon } from "@hugeicons/core-free-icons"
import { cn } from "@/lib/utils"
import ReactMarkdown from "react-markdown"
import { Button } from "@/components/ui/button"

interface Message {
  role: "user" | "ai"
  content: string
  plotly_json?: string | null
  charts?: string[]
  timestamp?: Date
}

interface User {
  id: number
  username: string
  email: string
}

interface ChatProps {
  user: User
  onLogout: () => void
  apiBase: string
}

export function Chat({ user, onLogout, apiBase }: ChatProps) {
  const [messages, setMessages] = React.useState<Message[]>([])
  const [input, setInput] = React.useState("")
  const [isLoading, setIsLoading] = React.useState(false)
  const [conversationId, setConversationId] = React.useState<string | null>(null)
  const messagesEndRef = React.useRef<HTMLDivElement>(null)

  // Scroll to bottom when messages change
  React.useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const sendMessage = async (query: string) => {
    if (!query.trim() || isLoading) return

    const userMessage: Message = {
      role: "user",
      content: query,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      const response = await fetch(`${apiBase}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
        body: JSON.stringify({
          query,
          conversation_id: conversationId,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      // Update conversation ID if provided
      if (data.conversation_id) {
        setConversationId(data.conversation_id)
      }

      const aiMessage: Message = {
        role: "ai",
        content: data.summary || data.error || "No response",
        plotly_json: data.plotly_json,
        charts: data.charts,
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, aiMessage])
    } catch (error) {
      const errorMessage: Message = {
        role: "ai",
        content: `Error: ${error instanceof Error ? error.message : "Failed to send message"}`,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    sendMessage(input)
  }

  // Suggested questions based on backend analysis types
  const suggestedQuestions = [
    "Who are my best customers?",
    "How are my sales trending over time?",
    "Which products are performing best?",
    "What are my customer segments?",
  ]

  return (
    <div className="flex h-screen w-full flex-col bg-background">
      {/* Header with title and logout */}
      <div className="border-b border-border px-4 py-4">
        <div className="mx-auto max-w-4xl flex items-center justify-between">
          <div className="flex items-center gap-3">
            <HugeiconsIcon 
              icon={ChartIcon} 
              strokeWidth={2} 
              className="size-7 text-primary" 
            />
            <h1 className="text-2xl font-bold text-foreground">
              AI Data Analyst
            </h1>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-sm text-muted-foreground">{user.email}</span>
            <Button variant="outline" size="sm" onClick={onLogout} className="cursor-pointer">
              Logout
            </Button>
          </div>
        </div>
      </div>
      {messages.length === 0 ? (
        // Centered layout when no messages
        <div className="flex h-full flex-col items-center justify-center px-4">
          <h1 className="mb-12 text-2xl font-medium text-foreground">
            Time to analyze
          </h1>
          <div className="w-full max-w-4xl">
            <form onSubmit={handleSubmit}>
              <InputGroup className="w-full h-14">
                <InputGroupInput
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask anything"
                  disabled={isLoading}
                  className="text-base h-full px-5"
                />
                <InputGroupAddon align="inline-end" className="pr-1.5">
                  <InputGroupButton
                    type="submit"
                    variant="default"
                    className="h-11 w-11 rounded-3xl p-0"
                    disabled={isLoading || !input.trim()}
                  >
                    <HugeiconsIcon icon={ArrowRight01Icon} strokeWidth={2} className="size-5" />
                  </InputGroupButton>
                </InputGroupAddon>
              </InputGroup>
            </form>
            {/* Suggested questions */}
            <div className="mt-8 grid grid-cols-2 gap-4 max-w-2xl mx-auto">
              {suggestedQuestions.map((question, index) => (
                <Button
                  key={index}
                  variant="outline"
                  size="sm"
                  onClick={() => sendMessage(question)}
                  disabled={isLoading}
                  className="text-sm whitespace-normal h-auto py-2 px-3 text-center cursor-pointer"
                >
                  {question}
                </Button>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <>
          {/* Main chat area */}
          <div className="flex-1 overflow-y-auto px-4 py-8">
            <div className="mx-auto max-w-4xl">
              <div className="space-y-6 py-4">
                {messages.map((message, index) => (
                  <MessageBubble key={index} message={message} />
                ))}
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="rounded-2xl bg-muted px-4 py-3">
                      <div className="flex gap-1">
                        <div className="h-2 w-2 animate-pulse rounded-full bg-muted-foreground" />
                        <div className="h-2 w-2 animate-pulse rounded-full bg-muted-foreground delay-75" />
                        <div className="h-2 w-2 animate-pulse rounded-full bg-muted-foreground delay-150" />
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            </div>
          </div>

          {/* Input bar */}
          <div className="bg-background px-4 py-6">
            <div className="mx-auto max-w-4xl w-full">
              <form onSubmit={handleSubmit}>
                <InputGroup className="w-full h-14">
                  <InputGroupInput
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask anything"
                    disabled={isLoading}
                    className="text-base h-full px-5"
                  />
                  <InputGroupAddon align="inline-end" className="pr-1.5">
                    <InputGroupButton
                      type="submit"
                      variant="default"
                      className="h-11 w-11 rounded-3xl p-0 cursor-pointer"
                      disabled={isLoading || !input.trim()}
                    >
                      <HugeiconsIcon icon={ArrowRight01Icon} strokeWidth={2} className="size-5" />
                    </InputGroupButton>
                  </InputGroupAddon>
                </InputGroup>
              </form>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user"

  return (
    <div
      className={cn(
        "flex w-full",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      <div
        className={cn(
          "max-w-[80%] rounded-2xl px-4 py-3",
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted text-foreground"
        )}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap text-sm leading-relaxed">
            {message.content}
          </p>
        ) : (
          <div className="text-sm leading-relaxed">
            <ReactMarkdown
              components={{
                p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                em: ({ children }) => <em className="italic">{children}</em>,
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        )}
        {message.plotly_json && (
          <ChartRenderer plotlyJson={message.plotly_json} />
        )}
        {message.charts && message.charts.length > 0 && (
          <div className="mt-4 space-y-4">
            {message.charts.map((chartJson, idx) => (
              <ChartRenderer key={idx} plotlyJson={chartJson} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function ChartRenderer({ plotlyJson }: { plotlyJson: string }) {
  const chartRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    if (!chartRef.current || !plotlyJson) return

    // Dynamically import plotly only when needed
    import("plotly.js-dist-min")
      .then((Plotly) => {
        try {
          const figure = JSON.parse(plotlyJson)
          // Apply dark theme to the layout
          const darkLayout = {
            ...figure.layout,
            paper_bgcolor: "rgba(0, 0, 0, 0)",
            plot_bgcolor: "rgba(0, 0, 0, 0)",
            font: {
              ...figure.layout?.font,
              color: "#e5e7eb",
            },
            xaxis: {
              ...figure.layout?.xaxis,
              gridcolor: "rgba(255, 255, 255, 0.1)",
              linecolor: "rgba(255, 255, 255, 0.2)",
            },
            yaxis: {
              ...figure.layout?.yaxis,
              gridcolor: "rgba(255, 255, 255, 0.1)",
              linecolor: "rgba(255, 255, 255, 0.2)",
            },
          }
          Plotly.newPlot(chartRef.current!, figure.data, darkLayout, {
            responsive: true,
            displayModeBar: true,
          })
        } catch (error) {
          console.error("Error rendering chart:", error)
        }
      })
      .catch((error) => {
        console.error("Error loading plotly:", error)
      })
  }, [plotlyJson])

  return <div ref={chartRef} className="mt-4 w-full" />
}
