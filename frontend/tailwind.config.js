/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        industrial: {
          bg: "#0B0F17",
          card: "#111827",
          cardHover: "#1F2937",
          border: "#1F2937",
          borderActive: "#374151",
          textMain: "#F9FAFB",
          textMuted: "#9CA3AF",
          normal: "#10B981",       // ISO 10816 Zone A/B Good (Emerald)
          warning: "#F59E0B",      // ISO 10816 Zone C Unsatisfactory (Amber)
          danger: "#EF4444",       // ISO 10816 Zone D Unacceptable (Red)
          accent: "#3B82F6",       // Ingestion / Primary (Cyan/Blue)
          copilot: "#8B5CF6",      // AI Copilot Purple
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
}
