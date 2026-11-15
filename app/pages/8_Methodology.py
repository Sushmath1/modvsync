"""Page 11: Methodology."""
import streamlit as st

from components import ui

ui.page_setup("Methodology", "Mathematical and scientific foundation of the analysis")

st.caption("Reference formulations. The backend implementation is authoritative. Parameter values in use are "
           "reported by the backend on the analysis pages.")


def method(title, equation, variables, why, interpretation, limitations):
    with st.expander(title, expanded=False):
        if equation:
            st.latex(equation)
        st.markdown(f"**Variables.** {variables}")
        st.markdown(f"**Why it is used.** {why}")
        st.markdown(f"**Interpretation.** {interpretation}")
        st.markdown(f"**Limitations.** {limitations}")


method("1. Haversine distance",
       r"d = 2R\arcsin\left(\sqrt{\sin^2\tfrac{\Delta\varphi}{2} + \cos\varphi_1\cos\varphi_2\sin^2\tfrac{\Delta\lambda}{2}}\right)",
       r"$\varphi$ latitude and $\lambda$ longitude in radians, $R \approx 6371$ km (mean Earth radius).",
       "Gives the great-circle distance between two detections on the Earth's surface.",
       "The ground distance between a MODIS and a VIIRS detection centre.",
       "Treats the Earth as a sphere, and measures between pixel centres, not between fire locations.")

method("2. Pixel footprint estimation",
       r"A = s \times t",
       r"$s$ scan size and $t$ track size of the pixel in km, as reported for each detection.",
       "Pixel size grows towards the swath edge, so each detection has its own spatial uncertainty.",
       "A larger footprint means less certainty about where within the pixel the fire is.",
       "A rectangular approximation of a pixel that is not exactly rectangular on the ground.")

method("3. Spatial association threshold",
       r"d_{MV} \le \tfrac{1}{2}\sqrt{s_M^2 + t_M^2} + \tfrac{1}{2}\sqrt{s_V^2 + t_V^2} + u",
       r"$d_{MV}$ MODIS–VIIRS distance, $s, t$ scan/track sizes, $u$ spatial uncertainty allowance (km).",
       "Two detections are spatial candidates only if their footprints could plausibly overlap.",
       "A pair within the threshold is a spatial candidate, not a confirmed same-fire event.",
       "The threshold depends on $u$ (see the sensitivity analysis), and ignores timing while dates are missing.")

method("4. Spatial similarity",
       r"S = \max\left(0,\ 1 - \frac{d_{MV}}{d_{thr}}\right)",
       r"$d_{thr}$ the pair's spatial threshold.",
       "Ranks candidate pairs from closest (S near 1) to the edge of the threshold (S near 0).",
       "A relative closeness score.",
       "Not a probability that the two detections are the same fire.")

method("5. Temporal association (when dates are available)",
       r"|t_M - t_V| \le \Delta t",
       r"$t_M, t_V$ acquisition timestamps, $\Delta t$ temporal window.",
       "Fires change over hours. Pairs should also be close in time to be spatiotemporal candidates.",
       "Would upgrade spatial candidates to spatiotemporal candidates.",
       "**Currently blocked:** the supplied datasets have no acquisition dates.")

method("6. DBSCAN clustering",
       r"N_\varepsilon(p) = \{q : d(p,q) \le \varepsilon\},\quad p \text{ is a core point if } |N_\varepsilon(p)| \ge \text{minPts}",
       r"$\varepsilon$ neighbourhood radius, minPts the minimum number of neighbours, $d$ haversine distance.",
       "Finds dense groups of detections without fixing the number of clusters, and labels isolated points as noise.",
       "A cluster is a spatial concentration of detections.",
       "Without dates, clusters are spatial groupings, not individual fire events. Results depend on $\\varepsilon$ and minPts.")

method("7. FRP statistical analysis",
       r"\bar{x} = \frac{1}{n}\sum_{i=1}^{n} \mathrm{FRP}_i,\qquad \tilde{x} = \mathrm{median}(\mathrm{FRP})",
       "FRP in MW per detection, summarised separately for each sensor.",
       "Describes the intensity distribution each sensor reports.",
       "Mean, median and maximum per sensor. The median is robust to a few very large fires.",
       "Sensors differ in pixel size and sensitivity, so FRP values are not directly comparable and are never summed across a pair.")

method("8. Sensitivity analysis",
       r"N(\tau) = \left|\{(M, V) : d_{MV} \le \tau\}\right|",
       r"$N(\tau)$ number of candidate pairs at distance threshold $\tau$.",
       "Shows how much the results depend on the chosen threshold.",
       "A curve that rises steeply means the association count is very sensitive to the threshold.",
       "Describes stability, not correctness.")
