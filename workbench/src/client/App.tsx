import { Toast } from "@base-ui-components/react/toast";

import { fetchEnvironment, useAsync } from "./api.ts";
import NavBar from "./components/NavBar.tsx";
import { Banner, Toasts } from "./components/ui.tsx";
import Dashboard from "./pages/Dashboard.tsx";
import RunPage from "./pages/RunPage.tsx";
import RunsPage from "./pages/RunsPage.tsx";
import { Link, Router, useRoute } from "./router.tsx";
import { ThemeProvider } from "./components/theme.tsx";

function Pages() {
  const route = useRoute();
  const environment = useAsync((signal) => fetchEnvironment(signal), [], null);

  return (
    <div className="app">
      <NavBar environment={environment.data} />
      <main className="shell">
        {route.name === "home" ? (
          <Dashboard environment={environment.data} reloadEnvironment={environment.reload} />
        ) : route.name === "runs" ? (
          <RunsPage />
        ) : route.name === "run" ? (
          <RunPage id={route.id} />
        ) : (
          <Banner tone="warning" title="No such page">
            <Link href="/">Back to the dashboard</Link>
          </Banner>
        )}
      </main>
      <Toasts />
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <Toast.Provider>
        <Router>
          <Pages />
        </Router>
      </Toast.Provider>
    </ThemeProvider>
  );
}
