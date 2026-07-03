# VizFold Gateway

Prototype for the VizFold Science Gateway.

## Structure

- `apps/web`: browser-based VizFold interface, runnable locally and deployable later as a science gateway frontend.
- `apps/api`: backend API service placeholder.
- `packages/schemas`: shared data contracts.
- `packages/adapters`: model adapter interfaces and implementations.
- `examples`: sample inputs and outputs.
- `docs`: architecture and implementation notes.

## Local Development

### Prerequisites

- Node.js 20 LTS or later
- npm
- Git

> **Recommended:** Use `nvm` (Linux/macOS/WSL) or `nvm-windows` (Windows) to manage Node.js versions.

Verify your installation:

```bash
node -v
npm -v
```

---

### Clone the repository

```bash
git clone ...
```

---

### Install dependencies

```bash
cd science-gateway/apps/web
npm install
```

---

### Configure environment variables

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

or simply duplicate the file manually.

---

### Configure Prisma

```bash
npx prisma generate
npx prisma migrate dev
```

---

### Start the application

```bash
npm run dev
```

The application will be available at:

```
http://localhost:3000
```