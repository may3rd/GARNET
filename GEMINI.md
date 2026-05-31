# GARNET Development Guidelines

## Package Manager: Bun

**ALWAYS use `bun` instead of `npm`, `npx`, or `yarn`.**

### Common Commands

| Task | Command |
|------|---------|
| Install dependencies | `bun install` |
| Add a package | `bun add <package>` |
| Add dev dependency | `bun add -d <package>` |
| Run scripts | `bun run <script>` |
| Run dev server | `bun run dev` |
| Build | `bun run build` |
| Execute binary | `bunx <binary>` (instead of `npx`) |
| Create React app | `bunx create-vite@latest` |

### Why Bun?

- 10-30x faster than npm
- Built-in TypeScript support
- Compatible with npm packages
- Faster dev server startup

### Project Setup with Bun

```bash
# Create new Vite + React project
bunx create-vite@latest frontend --template react-ts

# Navigate and install
cd frontend
bun install

# Add Shadcn/ui
bunx shadcn@latest init

# Add Tailwind
bun add -d tailwindcss postcss autoprefixer
bunx tailwindcss init -p
```

---

## Code Style

- Use TypeScript for all new code
- Use functional components with hooks
- Use Zustand for state management
- Follow Shadcn/ui patterns for components
- Use absolute imports (`@/components/...`)

---

## Git Workflow

- Branch from `main` for features
- Use conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`
- Squash merge to main
