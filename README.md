# infosys_ClimateScope

This Streamlit dashboard uses a custom purple-and-white theme and includes lightweight animations to improve the user experience.

## Setup

- Make sure you have the required packages installed:
  ```bash
  pip install -r requirements.txt
  # or at minimum:
  pip install streamlit pandas numpy plotly scipy streamlit-lottie
  ```

- The application picks up theme settings from `.streamlit/config.toml` (primary color set to purple `#800080`).
- If you want the dashboard to launch a browser window when started locally, ensure `headless = false` under `[server]` in that same config file.

- Animations are provided via CSS fade‑ins and a Lottie file; `streamlit-lottie` is required for the latter.

