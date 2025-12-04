import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pyvista import UnstructuredGrid
import pyvista as pv
from scipy.interpolate import griddata

# -------------------------------------------------
# 1. PASSWORD AUTHENTICATION
# -------------------------------------------------
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        # Checks whether a password entered by the user is correct.
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password in the session state for security
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Please enter the password:", 
        type="password", 
        on_change=password_entered, 
        key="password"
    )
    
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Password incorrect. Please try again.")

    return False

if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# -------------------------------------------------
# NPZ LOAD
# -------------------------------------------------
@st.cache_data
def load_npz_case(npz_filename: str):
    data = np.load(npz_filename)
    points = data["points"]
    cells = data["cells"]
    cell_types = data["cell_types"]
    T = data["T"] if "T" in data else None
    U = data["U"] if "U" in data else None
    mesh = UnstructuredGrid(cells, cell_types, points)
    return mesh, T, U
# -------------------------------------------------
# OUTER GEOMETRY (FROM VTK) LOAD
# -------------------------------------------------
@st.cache_data
def load_outer_geometry(vtk_path: str):
    """
    Read VTK file, extract outer surface and triangulate it.
    Returns coordinates and triangle indices for Plotly Mesh3d.
    """
    mesh = pv.read(vtk_path)
    
    # Outer surface
    surface = mesh.extract_surface()
    surface = surface.triangulate()  # ensure triangles

    pts = surface.points  # (N, 3)
    faces = surface.faces.reshape(-1, 4)  # [n_pts, i, j, k] for each face
    triangles = faces[:, 1:]              # drop n_pts (always 3 after triangulate)

    xg = pts[:, 0]
    yg = pts[:, 1]
    zg = pts[:, 2]
    ig = triangles[:, 0]
    jg = triangles[:, 1]
    kg = triangles[:, 2]

    return xg, yg, zg, ig, jg, kg
# -------------------------------------------------
# INTERPOLATION CACHING
# -------------------------------------------------
@st.cache_data
def interpolate_slice(axis1_s, axis2_s, f_s, grid_resolution):
    """Cache interpolation results for performance"""
    axis1_min, axis1_max = axis1_s.min(), axis1_s.max()
    axis2_min, axis2_max = axis2_s.min(), axis2_s.max()
    
    grid_axis1 = np.linspace(axis1_min, axis1_max, grid_resolution)
    grid_axis2 = np.linspace(axis2_min, axis2_max, grid_resolution)
    grid_axis1_mesh, grid_axis2_mesh = np.meshgrid(grid_axis1, grid_axis2)
    
    # First: Linear interpolation
    grid_field = griddata(
        (axis1_s, axis2_s), 
        f_s, 
        (grid_axis1_mesh, grid_axis2_mesh), 
        method='linear',
        fill_value=np.nan
    )
    
    # Fill NaN values with nearest neighbor
    nan_mask = np.isnan(grid_field)
    if nan_mask.any():
        grid_field_nearest = griddata(
            (axis1_s, axis2_s), 
            f_s, 
            (grid_axis1_mesh, grid_axis2_mesh), 
            method='nearest'
        )
        grid_field[nan_mask] = grid_field_nearest[nan_mask]
    
    return grid_axis1, grid_axis2, grid_field

# -------------------------------------------------
# STREAMLIT SETTINGS
# -------------------------------------------------
st.set_page_config(
    page_title="Thermal Digital Twin for Aalborg University (AAU) Pilot",
    layout="wide"
)

# Custom CSS (padding + sidebar + slider + cards)
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.5rem;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        margin-top: 0rem;
        padding-top: 0rem;
    }
    h1 {
        margin-top: 0rem;
        padding-top: 0rem;
    }
    .element-container {
        margin-top: 0rem;
    }

    /* Sidebar slider color (track + active range + handle border) */
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] > div {
        background: #dbeafe;  /* light blue track */
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
        background: #2563eb;  /* primary blue active part */
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
        background: #ffffff;
        border: 2px solid #2563eb;
        box-shadow: 0 0 0 1px rgba(37, 99, 235, 0.35);
    }

    /* Sidebar info card for total cells */
    .sidebar-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.12);
        border: 1px solid #e5e7eb;
        margin-top: 0.75rem;
        margin-bottom: 0.75rem;
        font-size: 13px;
    }
    .sidebar-card-title {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280;
        margin-bottom: 0.25rem;
        font-weight: 600;
    }
    .sidebar-card-metric {
        font-size: 13px;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Thermal Digital Twin for Aalborg University (AAU) Pilot")

mesh, T_field, U_field = load_npz_case("AAU/validationCase.npz")

centers = mesh.cell_centers().points
x = centers[:, 0]
y = centers[:, 1]
z = centers[:, 2]

total_cells = x.size

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.markdown(
    """
    <h2 style='text-align: center; font-size:20px; margin-bottom: 0.2rem;'>
        Thermal Digital Twin for Data Centres
    </h2>
    <p style='text-align: center; font-size:14px; margin-top: 0rem; color:#6b7280;'>
        Powered by D&S Tech
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# Tab selection
view_tab = st.sidebar.radio("", ["Thermal Digital Twin", "About"], index=0)

# Only show settings when in Results tab
if view_tab == "Thermal Digital Twin":
    field_choice = st.sidebar.selectbox("Field to Display", ["Temperature", "Airflow Velocity"])
mode = st.sidebar.selectbox("View Mode", ["3D Scatter", "2D Slice"]) if view_tab == "Thermal Digital Twin" else None

# Slice axis selection (only for 2D mode)
if view_tab == "Thermal Digital Twin" and mode == "2D Slice":
    slice_axis = st.sidebar.selectbox("Slice Axis", ["X", "Y", "Z"])
    
    # Slice thickness control: 1-5%
    thickness_percent = st.sidebar.slider(
        "Slice Thickness (%)", 
        1, 5, 3, 1,
        help="Thicker slice = more data points but less precise location"
    )
    
    # Grid resolution: 500-1000
    grid_resolution = st.sidebar.slider(
        "Grid Resolution", 
        500, 2000, 1000, 100, 
        help="Higher = sharper but slower. 700-800 recommended."
    )
else:
    slice_axis = None
    thickness_percent = 3
    grid_resolution = 700

# Point count slider: 10000-50000
if view_tab == "Thermal Digital Twin":
    max_points = st.sidebar.slider(
        "Maximum Points (3D only)", 
        min_value=10000, 
        max_value=50000, 
        value=25000,
        step=5000,
        help="More points = slower performance"
    )
else:
    max_points = 20000

# Modern "card" for total cells in sidebar
if view_tab == "Thermal Digital Twin":
    displayed_points = min(total_cells, max_points)
    st.sidebar.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-card-title">Grid Size</div>
            <div class="sidebar-card-metric">Total cells: {total_cells:,}</div>
            <div class="sidebar-card-metric">Displayed: {displayed_points:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Logo section at the bottom of sidebar
st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <p style='font-size: 12px; color:#6b7280; font-weight:600; margin-bottom:4px;'>
        Developed by D&STECH © 2025
    </p>
    <p style='font-size: 11px; color:#6b7280; line-height:1.4;'>
        This tool was developed as part of the Heatwise Project. The Heatwise Project has received funding from the European Union’s Horizon Europe research and innovation programme under Grant Agreement No 101138491 and the Swiss Secretariat for Education, Research, and Innovation (SERI) under contract No 23.00606.
    </p>
    """,
    unsafe_allow_html=True
)

logo_col1, logo_col2 = st.sidebar.columns(2)

with logo_col1:
    st.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <a href='https://heatwise.eu' target='_blank'>
            <img src='https://raw.githubusercontent.com/DSTECHNO/TDW/main/AAU/heatwise_logo.svg' 
                 alt='HEATWISE' style='width: 100%; max-width: 120px;'>
        </a>
    </div>
    """, unsafe_allow_html=True)

with logo_col2:
    st.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <a href='https://dstechs.net/' target='_blank'>
            <img src='https://raw.githubusercontent.com/DSTECHNO/TDW/main/AAU/dstech_logo.png' 
                 alt='D&S Tech' style='width: 100%; max-width: 120px;'>
        </a>
    </div>
    """, unsafe_allow_html=True)

# Field selection
if view_tab == "Thermal Digital Twin":
    if field_choice == "Temperature":
        field = T_field - 273.15  # Convert Kelvin to Celsius
        color_label = "T [°C]"
    else:
        field = np.linalg.norm(U_field, axis=1)
        color_label = "|U| [m/s]"

# -------------------------------------------------
# ABOUT PAGE
# -------------------------------------------------
if view_tab == "About":
    st.markdown("""
    ## AAU Data Centre Overview
    """)
    
    # AAU Building Image
    st.image(
        "https://raw.githubusercontent.com/DSTECHNO/TDW/main/AAU/aau.png",
        caption="AAU BUILD - Institut for Byggeri, By og Miljø",
        width=500
    )
    
    st.markdown("""
    ### Facility Information
    - **Location:** Aalborg University, Denmark
    - **Cooling Technology:** Air-cooled infrastructure
    
    ### IT Infrastructure
    - **Number of Server Racks:** 5 (Rack 2-6)
    - **Total IT Capacity:** 2.24 kW
    - **Rack Power Range:** 219 W - 1,000 W
    - **Configuration:** Variable thermal loads
    
    ### Cooling System
    - **Cooling Method:** Air-based cooling units
    - **Number of Cooling Units:** 2
    - **Total Cooling Capacity:** 2.27 kW
    - **Air Flow Rate:** 1.0 - 1.4 kg/s per unit
    
    ### Operating Conditions
    - **Ambient Temperature:** 21.9°C
    - **Inlet Air Temperature:** 21-23°C
    - **Outlet Air Temperature:** 23-25°C
    - **Rack Air Flow:** 0.033 - 0.3 kg/s
    
    ---
    
    <h3 style='font-style: italic;'>D&S Tech | Digital Twin Solutions</h3>
    <p style='font-size: 14px;'><a href='https://dstechs.net/' target='_blank'>https://dstechs.net/</a></p>
    <p style='font-size: 16px;'>Get Your Thermal Digital Twin. Contact us today: <strong>datacenter@dstechs.net</strong></p>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# RESULTS PAGE
# -------------------------------------------------
elif view_tab == "Thermal Digital Twin":
    # -------------------------------------------------
    # LAYOUT
    # -------------------------------------------------
    col1, col2 = st.columns([7, 3])

    with col1:
        # Determine visualization title based on field
        if field_choice == "Temperature":
            viz_title = "🌡️ Data Centre Thermal Map "
        else:
            viz_title = "💨 Airflow Velocity Field"

        st.subheader(viz_title)

        # Downsampling for 3D
        N = x.size
        if N > max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(N, size=max_points, replace=False)
            x_plot, y_plot, z_plot = x[idx], y[idx], z[idx]
            f_plot = field[idx]
        else:
            x_plot, y_plot, z_plot, f_plot = x, y, z, field

        cmin = float(field.min())
        cmax = float(field.max())

        # -----------------------------------------
        # 3D SCATTER
        # -----------------------------------------
        if mode == "3D Scatter":
            fig = go.Figure()

                        # --- OUTER GEOMETRY FROM VTK AS TRANSPARENT SHELL ---
            try:
                gx, gy, gz, gi, gj, gk = load_outer_geometry("AAU/validationCase.vtk")

                fig.add_trace(go.Mesh3d(
                    x=gx,
                    y=gy,
                    z=gz,
                    i=gi,
                    j=gj,
                    k=gk,
                    opacity=0.25,                 
                    color="gray",
                    name="DC Geometry",
                    showscale=False,
                    lighting=dict(
                        ambient=1.0,
                        diffuse=0.0,
                        specular=0.0,
                        roughness=0.0,
                        fresnel=0.0
                    ),
                    hoverinfo="skip"
                ))
            except Exception as e:
                st.warning(f"Outer geometry (validationCase.vtk) could not be loaded: {e}")  

            
            # --- 3D SCATTER OF FIELD ---
            fig.add_trace(go.Scatter3d(
                x=x_plot,
                y=y_plot,
                z=z_plot,
                mode="markers",
                marker=dict(
                    size=1.8,
                    color=f_plot,
                    colorscale="Turbo",
                    cmin=cmin,
                    cmax=cmax,
                    opacity=0.6,
                    colorbar=dict(
                        title=dict(
                            text=color_label,
                            font=dict(color="black", size=14)
                        ),
                        tickfont=dict(color="black", size=12),
                    ),
                ),
                hovertemplate='X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<br>' + 
                              color_label + ': %{marker.color:.3f}<extra></extra>'
            ))
            
 
            fig.update_layout(
                height=700,
                scene=dict(
                    xaxis_title="X [m]",
                    yaxis_title="Y [m]",
                    zaxis_title="Z [m]",
                    bgcolor="white",
                    aspectmode="data",
                ),
                scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor='white',
            )

            st.plotly_chart(fig, use_container_width=True)

        # -----------------------------------------
        # 2D SLICE - X, Y, or Z (INTERPOLATED)
        # -----------------------------------------
        else:  # mode == "2D Slice"
            # Select coordinates based on axis
            if slice_axis == "X":
                coord = x
                coord_min, coord_max = float(x.min()), float(x.max())
                coord_label = "X [m]"
                axis1, axis2 = y, z
                axis1_label, axis2_label = "Y [m]", "Z [m]"
            elif slice_axis == "Y":
                coord = y
                coord_min, coord_max = float(y.min()), float(y.max())
                coord_label = "Y [m]"
                axis1, axis2 = x, z
                axis1_label, axis2_label = "X [m]", "Z [m]"
            else:  # Z
                coord = z
                coord_min, coord_max = float(z.min()), float(z.max())
                coord_label = "Z [m]"
                axis1, axis2 = x, y
                axis1_label, axis2_label = "X [m]", "Y [m]"

            default_coord = 0.5 * (coord_min + coord_max)

            slice_coord = st.slider(
                f"{slice_axis}-slice location",
                min_value=coord_min,
                max_value=coord_max,
                value=default_coord,
                step=(coord_max - coord_min) / 200,
            )

            # Variable thickness based on user input
            thickness = (thickness_percent / 100.0) * (coord_max - coord_min)
            mask = np.abs(coord - slice_coord) <= thickness

            axis1_s = axis1[mask]
            axis2_s = axis2[mask]
            f_s = field[mask]

            if axis1_s.size > 0:
                st.caption(f"{axis1_s.size:,} points in slice (thickness: {thickness_percent}%) → Interpolating to {grid_resolution}x{grid_resolution} grid")

                try:
                    # Use cached interpolation with NaN filling
                    grid_axis1, grid_axis2, grid_field = interpolate_slice(
                        axis1_s, axis2_s, f_s, grid_resolution
                    )

                    # Heatmap plot
                    fig2 = go.Figure()
                    
                    fig2.add_trace(go.Heatmap(
                        x=grid_axis1,
                        y=grid_axis2,
                        z=grid_field,
                        colorscale="Turbo",
                        zmin=cmin,
                        zmax=cmax,
                        colorbar=dict(
                            title=dict(
                                text=color_label,
                                font=dict(color="black", size=14)
                            ),
                            tickfont=dict(color="black", size=12),
                        ),
                        hovertemplate=f'{axis1_label}: ' + '%{x:.3f}<br>' + 
                                      f'{axis2_label}: ' + '%{y:.3f}<br>' + 
                                      color_label + ': %{z:.3f}<extra></extra>'
                    ))

                    fig2.update_layout(
                        height=700,
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        margin=dict(l=0, r=0, t=40, b=0),
                        xaxis=dict(title=axis1_label, scaleanchor="y", scaleratio=1),
                        yaxis=dict(title=axis2_label),
                    )

                    st.plotly_chart(fig2, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Interpolation failed: {e}")
            else:
                st.warning(f"No points at this {slice_axis} location, adjust the slice.")

    # -------------------------------------------------
    # STATS
    # -------------------------------------------------
    with col2:
        st.markdown("<h3 style='font-size: 20px; font-weight: bold;'>📊 Summary Statistics</h3>", unsafe_allow_html=True)
        
        st.write("Minimum =", f"{field.min():.2f}")
        st.write("Maximum =", f"{field.max():.2f}")
        st.write("Mean T =", f"{field.mean():.2f}")
        st.write("Std. Dev.T =", f"{field.std():.2f}")
        
        # KPI Assessment Table
        st.markdown("---")
        
        # Create HTML table with custom styling - bigger font
        st.markdown("""
        <style>
        .kpi-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            margin-top: 10px;
            margin-bottom: 10px;
        }
        .kpi-table th {
            background-color: #2c3e50;
            color: white;
            padding: 10px 6px;
            text-align: left;
            font-weight: bold;
            font-size: 13px;
        }
        .kpi-table td {
            padding: 10px 6px;
            border-bottom: 1px solid #ddd;
            font-size: 13px;
        }
        .kpi-table tr:hover {
            background-color: #f5f5f5;
        }
        .powered-by {
            margin-top: 8px;
            margin-bottom: 2px;
            font-weight: bold;
            font-size: 20px;
        }
        .website-link {
            margin-top: 0px;
            font-size: 13px;
        }
        .kpi-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        </style>
        
        <div class="kpi-title">KPI Assessment for AAU Data Centre</div>
        
        <table class="kpi-table">
            <tr>
                <th>KPI</th>
                <th>Value</th>
                <th>Assessment</th>
            </tr>
            <tr>
                <td><strong>RCI<sub>HI</sub></strong></td>
                <td>205.80</td>
                <td>🔥 Hot-air recirculation</td>
            </tr>
            <tr>
                <td><strong>RCI<sub>LO</sub></strong></td>
                <td>174.20</td>
                <td>❄️ Cold-air bypass</td>
            </tr>
            <tr>
                <td><strong>RTI</strong></td>
                <td>46.11</td>
                <td>⚠️ Overcooling</td>
            </tr>
            <tr>
                <td><strong>RHI</strong></td>
                <td>0.78</td>
                <td>🔥 Moderate hot-air recirculation</td>
            </tr>
            <tr>
                <td><strong>RI</strong></td>
                <td>97.21</td>
                <td>🔥 Hot-air recirculation</td>
            </tr>
            <tr>
                <td><strong>CCI</strong></td>
                <td>1.72</td>
                <td>⚠️ Overcooling and inefficient airflow management</td>
            </tr>
        </table>
        
        <div class="powered-by">Powered by D&S Tech</div>
        <div class="website-link"><a href="https://dstechs.net/" target="_blank">https://dstechs.net/</a></div> """, unsafe_allow_html=True)
        
