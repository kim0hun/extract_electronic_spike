const cols = [
  ...Array.from(
    { length: 6 },
    (_, i) => `F03_${String(i + 1).padStart(2, "0")}`,
  ),
  ...Array.from(
    { length: 3 },
    (_, i) => `F05_${String(i + 1).padStart(2, "0")}`,
  ),
  ...Array.from(
    { length: 25 },
    (_, i) => `F10_${String(i + 1).padStart(2, "0")}`,
  ),
];

const isDev = process.env.NODE_ENV === "dev";

const config = {
  dev: {
    python: "D:\\dev\\conda_envs\\spike_mon\\python.exe",
    logPath: "D:\\dev\\lmsAI\\lms_mon\\logs",
  },
  prod: {
    python: "C:\\Users\\abc\\miniconda3\\envs\\lms_mon\\python.exe",
    logPath: "C:\\kamtec\\lmsAI\\lms_mon\\logs",
  },
};

const envConfig = isDev ? config.dev : config.prod;

module.exports = {
  apps: cols.map((col) => ({
    name: col,
    namespace: "LMS",
    script: "lms_mon.py",
    interpreter: envConfig.python,

    out_file: `${envConfig.logPath}\\${col}\\out.log`,
    error_file: `${envConfig.logPath}\\${col}\\err.log`,
    log_date_format: "YYYY-MM-DD HH:mm:ss",

    env: {
      NODE_ENV: isDev ? "dev" : "prod",
      COL: col,
    },
  })),
};
