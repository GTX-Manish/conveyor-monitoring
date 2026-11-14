/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: "var(--surface-1)", raised: "var(--surface-2)", line: "var(--border)",
        ink: "var(--text-primary)", ink2: "var(--text-secondary)", ink3: "var(--text-muted)",
      },
    },
  },
  plugins: [],
};
