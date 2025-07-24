#!/usr/bin/env node

const { spawn, execSync } = require('cross-spawn');
const path = require('path');
const fs = require('fs');
const which = require('which');

function getPythonVersion(pythonCmd) {
  try {
    const version = execSync(`${pythonCmd} --version`, { encoding: 'utf8' }).trim();
    const match = version.match(/Python (\d+)\.(\d+)\.(\d+)/);
    if (match) {
      return {
        major: parseInt(match[1]),
        minor: parseInt(match[2]),
        patch: parseInt(match[3]),
        full: `${match[1]}.${match[2]}.${match[3]}`
      };
    }
  } catch (e) {}
  return null;
}

async function findPython() {
  const projectRoot = path.join(__dirname, '..');
  const venvPath = path.join(projectRoot, '.venv');
  
  // First, check if there's a virtual environment
  if (fs.existsSync(venvPath)) {
    const venvPython = process.platform === 'win32'
      ? path.join(venvPath, 'Scripts', 'python.exe')
      : path.join(venvPath, 'bin', 'python');
      
    if (fs.existsSync(venvPython)) {
      const version = getPythonVersion(venvPython);
      if (version) {
        console.error(`Using virtual environment Python: ${version.full}`);
        return venvPython;
      }
    }
  }
  
  // Check for Poetry environment
  try {
    const poetryEnv = execSync('poetry env info --path', { 
      cwd: projectRoot,
      encoding: 'utf8',
      stdio: ['pipe', 'pipe', 'ignore']
    }).trim();
    
    if (poetryEnv) {
      const poetryPython = process.platform === 'win32'
        ? path.join(poetryEnv, 'Scripts', 'python.exe')
        : path.join(poetryEnv, 'bin', 'python');
        
      if (fs.existsSync(poetryPython)) {
        const version = getPythonVersion(poetryPython);
        if (version) {
          console.error(`Using Poetry environment Python: ${version.full}`);
          return poetryPython;
        }
      }
    }
  } catch (e) {
    // Poetry not available or no environment
  }
  
  // Check common Python installation paths directly
  const commonPythonPaths = [
    '/opt/homebrew/bin/python3.12',
    '/opt/homebrew/bin/python3.11',
    '/usr/local/bin/python3.12', 
    '/usr/local/bin/python3.11',
    '/usr/bin/python3.12',
    '/usr/bin/python3.11'
  ];
  
  for (const pythonPath of commonPythonPaths) {
    if (fs.existsSync(pythonPath)) {
      const version = getPythonVersion(pythonPath);
      if (version && version.major === 3 && version.minor >= 11 && version.minor < 13) {
        console.error(`Using Python at: ${pythonPath} (${version.full})`);
        return pythonPath;
      }
    }
  }
  
  // Fallback to system Python (with version check)
  const pythonCommands = ['python3.12', 'python3.11', 'python3', 'python'];
  
  for (const cmd of pythonCommands) {
    try {
      const pythonPath = await which(cmd);
      const version = getPythonVersion(pythonPath);
      if (version && version.major === 3 && version.minor >= 11 && version.minor < 13) {
        console.error(`Using system Python: ${cmd} (${version.full})`);
        return pythonPath;
      }
    } catch (e) {
      continue;
    }
  }
  
  throw new Error('Python 3.11 or 3.12 not found. Please ensure Python >=3.11,<3.13 is installed.');
}

async function main() {
  try {
    const projectRoot = path.join(__dirname, '..');
    const pythonPath = await findPython();
    
    console.error('Starting arxiv-mcp-server...');
    
    // Check if running with Poetry
    const usePoetry = pythonPath.includes('.venv') || pythonPath.includes('poetry');
    
    let proc;
    if (usePoetry && fs.existsSync(path.join(projectRoot, 'poetry.lock'))) {
      // Try to run with poetry first
      try {
        proc = spawn('poetry', ['run', 'arxiv-mcp-server'], {
          cwd: projectRoot,
          stdio: 'inherit',
          env: process.env
        });
      } catch (e) {
        // Fallback to direct Python execution
        proc = spawn(pythonPath, ['-m', 'arxiv_mcp_server.server'], {
          cwd: projectRoot,
          stdio: 'inherit',
          env: {
            ...process.env,
            PYTHONPATH: path.join(projectRoot, 'src')
          }
        });
      }
    } else {
      // Direct Python execution
      proc = spawn(pythonPath, ['-m', 'arxiv_mcp_server.server'], {
        cwd: projectRoot,
        stdio: 'inherit',
        env: {
          ...process.env,
          PYTHONPATH: path.join(projectRoot, 'src')
        }
      });
    }
    
    proc.on('error', (err) => {
      console.error('Failed to start server:', err.message);
      console.error('Please ensure all dependencies are installed:');
      console.error('  npm install -g arxiv-mcp-server');
      console.error('Or if installed locally:');
      console.error('  cd node_modules/arxiv-mcp-server && npm run postinstall');
      process.exit(1);
    });
    
    proc.on('exit', (code) => {
      process.exit(code || 0);
    });
    
    process.on('SIGINT', () => {
      proc.kill('SIGINT');
    });
    
    process.on('SIGTERM', () => {
      proc.kill('SIGTERM');
    });
    
  } catch (error) {
    console.error('Error:', error.message);
    console.error('\nTroubleshooting:');
    console.error('1. Ensure Python 3.11 or 3.12 is installed');
    console.error('2. Try reinstalling: npm install -g arxiv-mcp-server');
    console.error('3. Or use Poetry directly: cd /path/to/arxiv-mcp-server && poetry install');
    process.exit(1);
  }
}

main();