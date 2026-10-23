import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: 'tests/e2e',
  timeout: 120_000,
  webServer: {
    command: 'PORT=8123 node scripts/serve.js',
    port: 8123,
    reuseExistingServer: true,
  },
  use: {
    baseURL: 'http://localhost:8123',
    browserName: 'chromium',
    headless: true,
    launchOptions: {
      // CHROME_BIN lets a machine without Playwright's own browser use a local Chrome.
      ...(process.env.CHROME_BIN ? { executablePath: process.env.CHROME_BIN } : {}),
      args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
    },
  },
});
