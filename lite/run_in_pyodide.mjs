// Run notebooks headlessly in Pyodide under Node, the way the JupyterLite Pyodide kernel runs them in a browser.
//
// Usage: node lite/run_in_pyodide.mjs <contents-dir> [notebook.ipynb ...]
//
// <contents-dir> is the JupyterLite contents staged by tools/notebook_hooks.py (`python tools/notebook_hooks.py
// stage <dir>`): the runnable chapters' src/ files, with the install cell already prepended. With no notebook
// arguments, every notebook listed in <contents-dir>/runnable.txt runs. Each notebook gets a fresh interpreter,
// as a new kernel would: the contents are mounted at /drive, the working directory is the notebook's folder and
// '' is on sys.path (IPython's init_path), so `from case7 import dispatch` finds the sibling module. `%pip
// install` lines become micropip installs, as piplite does in the browser. Any failing cell fails the run.

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { loadPyodide } from 'pyodide';

const [contents, ...requested] = process.argv.slice(2);
if (!contents) {
  console.error('usage: node lite/run_in_pyodide.mjs <contents-dir> [notebook.ipynb ...]');
  process.exit(2);
}
const root = path.resolve(contents);
const notebooks = requested.length
  ? requested
  : readFileSync(path.join(root, 'runnable.txt'), 'utf8').split(/\r?\n/).filter(Boolean);
// Downloaded wheels are kept here between notebooks and between runs (CI caches the folder).
const packageCacheDir = process.env.PYODIDE_PACKAGE_CACHE || path.join(root, '..', 'pyodide-packages');

// `%pip install [--no-deps] [-q] pkg ...` -> the micropip call piplite makes for it.
function pipToPython(line) {
  const words = line.trim().split(/\s+/).slice(2);
  const packages = words.filter((w) => !w.startsWith('-'));
  const flags = words.filter((w) => w.startsWith('-'));
  for (const flag of flags) {
    if (!['--no-deps', '-q', '--quiet'].includes(flag)) throw new Error(`unsupported %pip flag ${flag}`);
  }
  const deps = flags.includes('--no-deps') ? 'False' : 'True';
  return `await __import__('micropip').install(${JSON.stringify(packages)}, deps=${deps})`;
}

function cellCode(source) {
  const text = Array.isArray(source) ? source.join('') : source;
  return text
    .split('\n')
    .map((line) => (/^\s*%pip\s+install\b/.test(line) ? pipToPython(line) : line))
    .join('\n');
}

async function runNotebook(relPath) {
  const nb = JSON.parse(readFileSync(path.join(root, relPath), 'utf8'));
  let output = '';
  const pyodide = await loadPyodide({
    packageCacheDir,
    stdout: (s) => (output += s + '\n'),
    stderr: (s) => (output += s + '\n'),
  });
  await pyodide.loadPackage('micropip', { messageCallback: () => {} });
  pyodide.FS.mkdirTree('/drive');
  pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root }, '/drive');
  const cwd = path.posix.join('/drive', path.posix.dirname(relPath.split(path.sep).join('/')));
  await pyodide.runPythonAsync(
    `import os, sys\nos.environ['MPLBACKEND'] = 'Agg'\nos.chdir(${JSON.stringify(cwd)})\nsys.path.insert(0, '')`,
  );
  const globals = pyodide.toPy({ __name__: '__main__' });
  let index = 0;
  for (const cell of nb.cells) {
    if (cell.cell_type !== 'code') continue;
    index += 1;
    const code = cellCode(cell.source);
    try {
      await pyodide.loadPackagesFromImports(code, { messageCallback: () => {} });
      await pyodide.runPythonAsync(code, { globals });
    } catch (error) {
      throw new Error(`code cell ${index} failed:\n${code}\n${output}\n${error.message ?? error}`);
    }
  }
  globals.destroy();
}

let failed = 0;
for (const relPath of notebooks) {
  const start = Date.now();
  try {
    await runNotebook(relPath);
    console.log(`PASS ${relPath} (${((Date.now() - start) / 1000).toFixed(1)} s)`);
  } catch (error) {
    failed += 1;
    console.log(`FAIL ${relPath}\n${error.message}`);
  }
}
console.log(`${notebooks.length - failed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
