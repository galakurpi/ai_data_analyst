import * as React from "react"
import { Chat } from "@/components/Chat"
import { Login } from "@/components/Login"

// Use relative URL in production, localhost in dev
const API_BASE = import.meta.env.PROD ? "" : "http://localhost:8000"

interface User {
  id: number
  username: string
  email: string
}

export function App() {
  const [user, setUser] = React.useState<User | null>(null)
  const [isChecking, setIsChecking] = React.useState(true)

  // Check auth on mount
  React.useEffect(() => {
    const checkAuth = async () => {
      try {
        const response = await fetch(`${API_BASE}/auth/check`, {
          credentials: "include",
        })
        const data = await response.json()
        if (data.authenticated) {
          setUser(data.user)
        }
      } catch (err) {
        console.error("Auth check failed:", err)
      } finally {
        setIsChecking(false)
      }
    }
    checkAuth()
  }, [])

  const handleLogin = (user: User) => {
    setUser(user)
  }

  const handleLogout = async () => {
    try {
      await fetch(`${API_BASE}/auth/logout`, {
        method: "POST",
        credentials: "include",
      })
    } catch (err) {
      console.error("Logout failed:", err)
    }
    setUser(null)
  }

  if (isChecking) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <div className="text-muted-foreground">Loading...</div>
      </div>
    )
  }

  if (!user) {
    return <Login onLogin={handleLogin} apiBase={API_BASE} />
  }

  return <Chat user={user} onLogout={handleLogout} apiBase={API_BASE} />
}

export default App
