import { defineConfig } from 'vitest/config';

// `vite build` builds the library (https://vite.dev/guide/build#library-mode);
// `vite build --mode example` builds the example page as a static site.
export default defineConfig(({ mode }) => ({
  build: mode === 'example'
    ? { outDir: 'dist-example' }
    : {
        lib: { entry: 'src/index.ts', formats: ['es'], fileName: 'index' },
        // tsc writes the declarations into dist first.
        emptyOutDir: false,
        sourcemap: true,
      },
  test: {
    environment: 'node',
    // A software adapter (lavapipe in CI) with a cold shader cache takes
    // seconds to compile a test's pipelines.
    testTimeout: 60_000,
  },
}));
