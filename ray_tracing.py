import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Arc, FancyArrowPatch
import numpy as np

# Configure standard Matplotlib styling
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "legend.fontsize": 11,
        "figure.dpi": 100,
        "figure.facecolor": "black",
        "axes.facecolor": "#00001A",
        "text.color": "white",
        "axes.labelcolor": "white",
        "axes.titlecolor": "white",
        "axes.edgecolor": "white",
    }
)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

_SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = _SCRIPT_DIR / "OUTPUTS"
PLOT_DIR = OUTPUT_DIR / "PLOTS"
ANIMATION_DIR = OUTPUT_DIR / "ANIMATIONS"

for folder in [OUTPUT_DIR, PLOT_DIR, ANIMATION_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


class CavityParameters:
    """Simple class to store cavity configuration parameters."""

    def __init__(
        self,
        R1,
        R2,
        L,
        g1,
        g2,
        stability_product,
        is_stable,
        is_confocal,
        is_symmetric,
        mirror1_type,
        mirror2_type,
        cavity_config,
        mirror_description,
    ):
        self.R1 = R1
        self.R2 = R2
        self.L = L
        self.g1 = g1
        self.g2 = g2
        self.stability_product = stability_product
        self.is_stable = is_stable
        self.is_confocal = is_confocal
        self.is_symmetric = is_symmetric
        self.mirror1_type = mirror1_type
        self.mirror2_type = mirror2_type
        self.cavity_config = cavity_config
        self.mirror_description = mirror_description


class CavityRayTracing:
    """
    Parent class for ray tracing in optical cavities with concave and/or convex mirrors.

    This base class provides core functionality for general cavity ray tracing, including:
    - Mirror geometry and intersection calculations for both concave (R < 0) and convex (R > 0) mirrors
    - Ray transfer matrix operations (ABCD matrix method)
    - Stability parameter calculation and analysis
    - Parameter validation and error handling
    - Basic static visualization and figure export utilities

    This class supports all cavity configurations: Concave-Concave, Convex-Convex, and Concave-Convex.

    Key Features:
    - Single ray path visualization with dynamic titles and annotations
    - Proper sign handling for concave (negative R) vs convex (positive R) mirrors
    - Stability parameter display (g1, g2) and product checks
    - Interactive CLI parameter input
    """

    def __init__(self, R1=-80.0, R2=-80.0, L=70.0):
        """
        Initialize the cavity ray tracer.

        Parameters
        ----------
        R1 : float
            Radius of curvature for mirror 1 (negative for concave, positive for convex)
            Default: -80.0 cm
        R2 : float
            Radius of curvature for mirror 2 (negative for concave, positive for convex)
            Default: -80.0 cm
        L : float
            Cavity length (distance between mirror vertices)
            Default: 70.0 cm

        Notes
        -----
        - For concave mirrors, use negative R values (e.g., R1 = -80.0)
        - For convex mirrors, use positive R values (e.g., R1 = 80.0)
        - Stability requires: 0 <= g1*g2 <= 1, where g = 1 + L/R
        - Confocal condition: L = |R1| = |R2| (only for concave-concave cavities)

        Raises
        ------
        ValueError
            If parameters are invalid
        """
        self._validate_cavity_parameters(R1, R2, L)
        self.R1 = R1
        self.R2 = R2
        self.L = L
        self._calculate_cavity_properties()

        # Theme color mapping based on ray color
        self.color_dict = {
            "red": {
                "ax_face": "#110000",
                "timer_bg": "#470000",
                "params_bg": "#FFB8B8",
            },
            "blue": {
                "ax_face": "#00001A",
                "timer_bg": "#000047",
                "params_bg": "#B8B8FF",
            },
            "green": {
                "ax_face": "#001100",
                "timer_bg": "#004700",
                "params_bg": "#B8FFB8",
            },
            "orange": {
                "ax_face": "#090500",
                "timer_bg": "#471F00",
                "params_bg": "#FFD8B8",
            },
            "purple": {
                "ax_face": "#0C000C",
                "timer_bg": "#470047",
                "params_bg": "#FFB8FF",
            },
        }

    def _validate_cavity_parameters(self, R1, R2, L):
        """
        Validate cavity parameters.

        Parameters
        ----------
        R1, R2 : float
            Radii of curvature
        L : float
            Cavity length

        Raises
        ------
        ValueError
            If any parameter is invalid
        TypeError
            If parameters are not numeric
        """
        # Type checking
        if not all(isinstance(x, (int, float)) for x in [R1, R2, L]):
            raise TypeError("R1, R2, and L must be numeric values")

        # Value validation
        if R1 == 0:
            raise ValueError("R1 cannot be zero")
        if R2 == 0:
            raise ValueError("R2 cannot be zero")
        if L <= 0:
            raise ValueError("Cavity length L must be positive")

        # Physical constraints (check against absolute radius)
        if abs(R1) < L / 10:
            raise ValueError("R1 appears too small for the cavity length")
        if abs(R2) < L / 10:
            raise ValueError("R2 appears too small for the cavity length")

    def _calculate_cavity_properties(self):
        """Calculate and store cavity properties (g-parameters, stability, etc.)."""
        self.g1 = 1 + self.L / self.R1
        self.g2 = 1 + self.L / self.R2
        self.stability_product = self.g1 * self.g2
        self.is_stable = 0 <= self.stability_product <= 1

        # Mirror types determination
        self.mirror1_type = "Concave" if self.R1 < 0 else "Convex"
        self.mirror2_type = "Concave" if self.R2 < 0 else "Convex"

        # Cavity configuration determination based on mirror types
        if self.R1 < 0 and self.R2 < 0:
            self.cavity_config = "Concave-Concave"
        elif self.R1 > 0 and self.R2 > 0:
            self.cavity_config = "Convex-Convex"
        elif self.R1 < 0 and self.R2 > 0:
            self.cavity_config = "Concave-Convex"
        else:
            self.cavity_config = "Convex-Concave"

        # Determination of mirror description string for titles
        if self.mirror1_type == self.mirror2_type:
            self.mirror_description = f"{self.mirror1_type}"
        else:
            self.mirror_description = f"{self.mirror1_type}-{self.mirror2_type}"

        # Confocal cavity checking (only applies to concave-concave with L = |R1| = |R2|)
        self.is_confocal = (
            self.R1 < 0
            and self.R2 < 0
            and abs(self.L - abs(self.R1)) < 1e-6
            and abs(abs(self.R1) - abs(self.R2)) < 1e-6
        )

        # Symmetric cavity checking (radii are identical, including sign)
        self.is_symmetric = abs(self.R1 - self.R2) < 1e-6

        # Mirror centers for plotting
        self.left_mirror_center_x = -self.L / 2 - self.R1
        self.right_mirror_center_x = self.L / 2 + self.R2

        # Focal points
        x_pole_L = -self.L / 2.0
        x_pole_R = +self.L / 2.0
        self.F1x = x_pole_L - self.R1 * 0.5
        self.F2x = x_pole_R + self.R2 * 0.5

        # Centers of curvature
        self.C1x = self.left_mirror_center_x
        self.C2x = self.right_mirror_center_x

    def get_cavity_parameters(self):
        """
        Get cavity parameters and stability information.

        Returns
        -------
        CavityParameters
            Object containing all cavity parameters
        """
        return CavityParameters(
            R1=self.R1,
            R2=self.R2,
            L=self.L,
            g1=self.g1,
            g2=self.g2,
            stability_product=self.stability_product,
            is_stable=self.is_stable,
            is_confocal=self.is_confocal,
            is_symmetric=self.is_symmetric,
            mirror1_type=self.mirror1_type,
            mirror2_type=self.mirror2_type,
            cavity_config=self.cavity_config,
            mirror_description=self.mirror_description,
        )

    def _get_x_on_mirror(self, y, R, center_x, side):
        """
        Calculate the x-coordinate on a spherical mirror for a given y.

        Parameters
        ----------
        y : float
            Y-coordinate
        R : float
            Radius of curvature
        center_x : float
            X-coordinate of mirror center
        side : str
            'left' or 'right' indicating which mirror

        Returns
        -------
        float
            X-coordinate on mirror surface, or np.nan if invalid
        """
        if abs(y) > abs(R):
            return np.nan

        discriminant = R**2 - y**2
        if discriminant < 0:
            return np.nan

        if side == "left":
            return center_x + np.sign(R) * np.sqrt(discriminant)
        elif side == "right":
            return center_x - np.sign(R) * np.sqrt(discriminant)
        else:
            return np.nan

    def _get_mirror_arcs(self, arc_angle):
        """Helper to create left and right mirror Arc patches properly oriented for concave or convex mirrors."""
        if self.R1 < 0:
            theta1_left, theta2_left = 180 - arc_angle, 180 + arc_angle
        else:
            theta1_left, theta2_left = -arc_angle, arc_angle

        if self.R2 < 0:
            theta1_right, theta2_right = -arc_angle, arc_angle
        else:
            theta1_right, theta2_right = 180 - arc_angle, 180 + arc_angle

        left_mirror = Arc(
            (self.left_mirror_center_x, 0),
            2 * abs(self.R1),
            2 * abs(self.R1),
            angle=0,
            theta1=theta1_left,
            theta2=theta2_left,
            color="gray",
            lw=6,
            zorder=5,
        )
        right_mirror = Arc(
            (self.right_mirror_center_x, 0),
            2 * abs(self.R2),
            2 * abs(self.R2),
            angle=0,
            theta1=theta1_right,
            theta2=theta2_right,
            color="gray",
            lw=6,
            zorder=5,
        )
        return left_mirror, right_mirror

    def _validate_ray_parameters(self, y0_initial, theta0_initial_deg, N_round_trips):
        """
        Validate ray tracing parameters.

        Parameters
        ----------
        y0_initial : float
            Initial ray height
        theta0_initial_deg : float
            Initial angle in degrees
        N_round_trips : int
            Number of round trips

        Raises
        ------
        ValueError
            If parameters are invalid
        TypeError
            If parameters have wrong type
        """
        if not isinstance(N_round_trips, int):
            raise TypeError("N_round_trips must be an integer")

        if N_round_trips < 1:
            raise ValueError("N_round_trips must be at least 1")

        if N_round_trips > 100:
            raise ValueError("N_round_trips too large (max 100)")

        if not isinstance(y0_initial, (int, float)):
            raise TypeError("y0_initial must be numeric")

        if not isinstance(theta0_initial_deg, (int, float)):
            raise TypeError("theta0_initial_deg must be numeric")

        if abs(y0_initial) > abs(self.R1) * 0.9:
            raise ValueError(f"Initial height too large (max {abs(self.R1) * 0.9:.1f})")

        if abs(theta0_initial_deg) > 45:
            raise ValueError("Initial angle too large (max ±45°)")

    def trace_ray(self, y0_initial=15.0, theta0_initial_deg=0.0, N_round_trips=2):
        """
        Perform ray tracing through the cavity.

        Parameters
        ----------
        y0_initial : float
            Initial height of ray from optical axis
        theta0_initial_deg : float
            Initial angle in degrees
        N_round_trips : int
            Number of round trips to simulate

        Returns
        -------
        ray_segments : list
            List of ray path segments, each segment is a list of (x, y) tuples
        final_state : numpy.ndarray
            Final state vector [y, theta] after all round trips

        Raises
        ------
        ValueError
            If ray parameters are invalid
        """
        self._validate_ray_parameters(y0_initial, theta0_initial_deg, N_round_trips)

        # Convert angle to radians
        theta0_initial = np.deg2rad(theta0_initial_deg)

        # Transfer matrices
        M_prop = np.array([[1, self.L], [0, 1]])
        M_refl_1 = np.array([[1, 0], [2 / self.R1, 1]])
        M_refl_2 = np.array([[1, 0], [2 / self.R2, 1]])

        # Ray tracing initialization
        ray_segments = []
        current_segment = []

        # Initial state vector [y, theta]
        y_theta_vec = np.array([[y0_initial], [theta0_initial]])

        # Initial point on Mirror 1
        x0 = self._get_x_on_mirror(
            y0_initial, self.R1, self.left_mirror_center_x, "left"
        )

        if np.isnan(x0):
            raise ValueError("Initial ray position is outside mirror aperture")

        current_segment.append((x0, y0_initial))

        # Ray tracing
        for i in range(N_round_trips):
            # Propagation from Mirror 1 to Mirror 2
            y_theta_vec = M_prop @ y_theta_vec
            y_at_M2 = y_theta_vec[0, 0]
            x_at_M2 = self._get_x_on_mirror(
                y_at_M2, self.R2, self.right_mirror_center_x, "right"
            )

            if np.isnan(x_at_M2):
                print(
                    f"\n[Notice] Ray escaped at Mirror 2 during round trip {i + 1} "
                    f"(height |y| = {abs(y_at_M2):.2f} cm > aperture |R2| = {abs(self.R2):.2f} cm)."
                )
                current_segment.append((self.L / 2, y_at_M2))
                ray_segments.append(current_segment.copy())
                break

            current_segment.append((x_at_M2, y_at_M2))
            ray_segments.append(current_segment.copy())
            current_segment = [(x_at_M2, y_at_M2)]

            # Reflection from Mirror 2
            y_theta_vec = M_refl_2 @ y_theta_vec

            # Propagation from Mirror 2 back to Mirror 1
            y_theta_vec = M_prop @ y_theta_vec
            y_at_M1 = y_theta_vec[0, 0]
            x_at_M1 = self._get_x_on_mirror(
                y_at_M1, self.R1, self.left_mirror_center_x, "left"
            )

            if np.isnan(x_at_M1):
                print(
                    f"\n[Notice] Ray escaped at Mirror 1 during round trip {i + 1} "
                    f"(height |y| = {abs(y_at_M1):.2f} cm > aperture |R1| = {abs(self.R1):.2f} cm)."
                )
                current_segment.append((-self.L / 2, y_at_M1))
                ray_segments.append(current_segment.copy())
                break

            current_segment.append((x_at_M1, y_at_M1))
            ray_segments.append(current_segment.copy())
            current_segment = [(x_at_M1, y_at_M1)]

            # Reflection from Mirror 1
            y_theta_vec = M_refl_1 @ y_theta_vec

        return ray_segments, y_theta_vec

    def print_info(self, y0_initial, theta0_initial_deg, N_round_trips):
        """Print detailed cavity information and simulation parameters."""
        params = self.get_cavity_parameters()

        print("=" * 50)
        print("CAVITY INFORMATION")
        print("=" * 50)
        print(
            f"Mirror 1 radius (R1):        {params.R1:.2f} cm ({params.mirror1_type})"
        )
        print(
            f"Mirror 2 radius (R2):        {params.R2:.2f} cm ({params.mirror2_type})"
        )
        print(f"Cavity length (L):           {params.L:.2f} cm")
        print("-" * 50)
        print(f"Stability parameter g1:      {params.g1:.4f}")
        print(f"Stability parameter g2:      {params.g2:.4f}")
        print(f"Stability product (g1*g2):   {params.stability_product:.4f}")
        print("-" * 50)

        cavity_symmetry = "Symmetric" if params.is_symmetric else "Asymmetric"
        if params.is_confocal:
            cavity_type = "CONFOCAL"
        elif params.is_symmetric:
            cavity_type = f"Symmetric {params.mirror_description}"
        else:
            cavity_type = f"Asymmetric {params.mirror_description}"
        print(
            f"Cavity configuration:        {params.cavity_config} ({cavity_symmetry})"
        )
        print(f"Cavity type:                 {cavity_type}")
        print(
            f"Stability status:            {'STABLE' if params.is_stable else 'UNSTABLE'}"
        )
        if params.is_stable:
            print("[+] Cavity is stable for beam propagation")
        else:
            print("[-] Cavity is unstable - beam will diverge")
        print("=" * 50)
        print("SIMULATION PARAMETERS")
        print("=" * 50)
        print(f"Initial ray height (y0):     {y0_initial:.2f} cm")
        print(f"Initial angle (theta0):      {theta0_initial_deg:.2f} deg")
        print(f"Number of round trips:       {N_round_trips}")
        print("=" * 50)

    def trace_single_ray(
        self, y0_initial=15.0, theta0_initial_deg=0.0, N_round_trips=50
    ):
        """
        Trace a single ray through the cavity.

        This method uses the parent class's trace_ray method but formats
        the output as a single continuous path rather than segments.

        Parameters
        ----------
        y0_initial : float
            Initial height of ray from optical axis (cm)
        theta0_initial_deg : float
            Initial angle in degrees
        N_round_trips : int
            Number of round trips to simulate

        Returns
        -------
        ray_path : list of tuples
            List of (x, y) coordinates along the ray path
        final_state : numpy.ndarray
            Final state vector [y, theta] after all round trips

        Raises
        ------
        ValueError
            If ray escapes the cavity or parameters are invalid
        """
        ray_segments, final_state = self.trace_ray(
            y0_initial, theta0_initial_deg, N_round_trips
        )

        ray_path = []
        for i, segment in enumerate(ray_segments):
            if i == 0:
                ray_path.extend(segment)
            else:
                ray_path.extend(segment[1:])

        return ray_path, final_state

    def visualize_single_ray(
        self,
        y0_initial=15.0,
        theta0_initial_deg=0.0,
        N_round_trips=50,
        arc_angle=25.0,
        ray_color="red",
        save_figure=False,
        filename=None,
    ):
        """
        Visualize single ray tracing through the optical cavity.

        Creates a detailed plot showing:
        - Mirror geometry with arcs (oriented for concave or convex)
        - Optical axis
        - Ray path with markers
        - Design parameters annotation
        - Cavity stability status

        Parameters
        ----------
        y0_initial : float
            Initial height of ray from optical axis
        theta0_initial_deg : float
            Initial angle in degrees
        N_round_trips : int
            Number of round trips to simulate
        arc_angle : float
            Angle for arc representation of mirrors (0-90 degrees)
        ray_color : str
            Color of the ray ('red', 'blue', 'green', 'orange', 'purple')
        save_figure : bool
            Whether to save the figure as PNG
        filename : str or None
            Filename for saved figure (auto-generated if None)

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The axes object

        Raises
        ------
        ValueError
            If parameters are invalid or ray escapes cavity
        """
        if not 0 < arc_angle < 90:
            raise ValueError("arc_angle must be between 0 and 90 degrees")

        try:
            ray_path, final_state = self.trace_single_ray(
                y0_initial, theta0_initial_deg, N_round_trips
            )
        except Exception as e:
            raise ValueError(f"Ray tracing failed: {str(e)}")

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.subplots_adjust(left=0.02, right=0.8, top=0.94, bottom=0.02)
        ax.set_facecolor(self.color_dict[ray_color]["ax_face"])

        left_mirror, right_mirror = self._get_mirror_arcs(arc_angle)
        ax.add_patch(left_mirror)
        ax.add_patch(right_mirror)

        # Optical axis
        ax.plot(
            [-self.L / 2, self.L / 2], [0, 0], color="black", lw=1, ls="--", zorder=1
        )

        # Plotting ray path
        path_x, path_y = zip(*ray_path)
        ax.plot(
            path_x,
            path_y,
            "-o",
            color=ray_color,
            markersize=3,
            label="Ray Path",
            zorder=3,
        )

        # Parameter box
        param_text = (
            f"Design Parameters\n"
            f"{'-' * 23}\n"
            f"$R_1$ = {self.R1:.1f} cm ({self.mirror1_type})\n"
            f"$R_2$ = {self.R2:.1f} cm ({self.mirror2_type})\n"
            f"$L$ = {self.L:.1f} cm\n"
            f"$g_1$ = {self.g1:.3f}\n"
            f"$g_2$ = {self.g2:.3f}\n"
            f"$g_1 \\times g_2$ = {self.stability_product:.3f}\n"
            f"Stability: {'STABLE' if self.is_stable else 'UNSTABLE'}\n"
            f"Initial $\\theta$ = {theta0_initial_deg:.1f}°\n"
            f"Round trips: {N_round_trips}"
        )

        ax.text(
            1.02,
            0.98,
            param_text,
            transform=ax.transAxes,
            va="top",
            bbox=dict(
                boxstyle="round",
                facecolor=self.color_dict[ray_color]["params_bg"],
                alpha=0.4,
            ),
            fontsize=10,
            fontfamily="monospace",
        )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        if self.is_confocal:
            title_str = "Confocal"
        elif self.is_symmetric:
            title_str = f"Symmetric {self.mirror_description}"
        else:
            title_str = f"Asymmetric {self.mirror_description}"
        ax.set_title(f"Single Ray Tracing in {title_str} Cavity", pad=10)
        ax.legend(loc="upper center")

        if save_figure:
            self._save_figure(fig, filename, theta0_initial_deg, ray_color)

        plt.show()

        return fig, ax

    def _save_figure(self, fig, filename, theta0_initial_deg, ray_color):
        """
        Save figure to PNG file.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save
        filename : str or None
            Base filename (extension appended automatically; auto-generated if None)
        theta0_initial_deg : float
            Initial angle (for auto-generated filename)
        ray_color : str
            Color of the ray (for auto-generated filename metadata)
        """
        if filename is None:
            config_prefix = self.cavity_config.lower().replace("-", "_")
            sym_prefix = "symmetric" if self.is_symmetric else "asymmetric"
            metadata = f"R1_{self.R1}_R2_{self.R2}_L_{self.L}_theta_{theta0_initial_deg}_rc_{ray_color}"
            filename = f"{sym_prefix}_{config_prefix}_cavity_{metadata}_{timestamp}"

        # Ensure .png extension
        base = Path(filename).stem if str(filename).endswith(".png") else filename
        filepath = PLOT_DIR / f"{base}.png"

        try:
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Figure saved successfully to: {filepath}")
        except Exception as e:
            print(f"Error saving figure: {e}")

    # -------------------------------------------------------------------------
    # Animation
    # -------------------------------------------------------------------------

    def _interpolate_segments(self, segments, points_per_segment):
        """
        Interpolate ray segments for smooth animation.

        Parameters
        ----------
        segments : list
            List of ray path segments
        points_per_segment : int
            Number of interpolation points per segment

        Returns
        -------
        all_points : list
            List of (x, y) coordinates for all interpolated points
        seg_ids : list
            Segment ID for each point
        pt_idx_in_seg : list
            Point index within each segment
        npps : int
            Actual number of points per segment used
        """
        all_points = []
        seg_ids = []
        pt_idx_in_seg = []
        npps = int(max(2, points_per_segment))

        for s_idx, seg in enumerate(segments):
            if len(seg) < 2:
                continue
            (x0, y0), (x1, y1) = seg[0], seg[1]
            tvals = np.linspace(0.0, 1.0, npps)
            for j, t in enumerate(tvals):
                all_points.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
                seg_ids.append(s_idx)
                pt_idx_in_seg.append(j)

        return all_points, seg_ids, pt_idx_in_seg, npps

    def animate_cavity(
        self,
        y0_initial=15.0,
        theta0_initial_deg=0.0,
        N_round_trips=50,
        arc_angle=25.0,
        fps=30,
        points_per_segment=5,
        ray_color="red",
        save_animation=False,
        filename=None,
        anim_format="gif",
    ):
        """
        Create animated visualization of ray tracing through the cavity.

        Parameters
        ----------
        y0_initial : float
            Initial ray height from optical axis (cm)
        theta0_initial_deg : float
            Initial angle in degrees
        N_round_trips : int
            Number of round trips to simulate
        arc_angle : float
            Angle for arc representation of mirrors (0-90 degrees)
        fps : int
            Frames per second for animation
        points_per_segment : int
            Number of interpolation points per ray segment
        ray_color : str
            Color of the ray ('red', 'blue', 'green', 'orange', 'purple')
        save_animation : bool
            Whether to save the animation
        filename : str or None
            Base filename for saved animation (extension appended automatically)
        anim_format : str
            Animation format: 'gif' or 'mp4'

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The axes object
        anim : matplotlib.animation.FuncAnimation
            The animation object
        """
        # Ray Tracing and Validation
        ray_segments, _ = self.trace_ray(y0_initial, theta0_initial_deg, N_round_trips)

        # Interpolate segments for smooth animation
        all_points, seg_ids, pt_idx_in_seg, npps = self._interpolate_segments(
            ray_segments, points_per_segment
        )

        # Figure and axes set up
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.subplots_adjust(left=0.02, right=0.8, top=0.94, bottom=0.02)
        ax.set_facecolor(self.color_dict[ray_color]["ax_face"])

        # Mirrors
        left_mirror, right_mirror = self._get_mirror_arcs(arc_angle)
        ax.add_patch(left_mirror)
        ax.add_patch(right_mirror)

        # Optical axis
        ax.plot(
            [-self.L / 2, self.L / 2],
            [0, 0],
            color="grey",
            lw=1,
            ls="--",
            zorder=1,
            alpha=0.5,
        )

        (ray_line,) = ax.plot(
            [], [], "-", linewidth=2, color=ray_color, zorder=3, alpha=0.85
        )
        arrow_patch = None

        if self.is_confocal:
            title_str = "Confocal"
        elif self.is_symmetric:
            title_str = f"Symmetric {self.mirror_description}"
        else:
            title_str = f"Asymmetric {self.mirror_description}"

        # Parameter text box
        param_text = (
            f"Design Parameters\n"
            f"{'-' * 23}\n"
            f"$R_1$ = {self.R1:.1f} cm ({self.mirror1_type})\n"
            f"$R_2$ = {self.R2:.1f} cm ({self.mirror2_type})\n"
            f"$L$ = {self.L:.1f} cm\n"
            f"$g_1$ = {self.g1:.3f}\n"
            f"$g_2$ = {self.g2:.3f}\n"
            f"$g_1 \\times g_2$ = {self.stability_product:.3f}\n"
            f"Stability: {'STABLE' if self.is_stable else 'UNSTABLE'}\n"
            f"$\\theta_0$ = {theta0_initial_deg:.1f}\u00b0\n"
            f"Round trips = {N_round_trips}"
        )
        ax.text(
            1.02,
            0.98,
            param_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="round",
                facecolor=self.color_dict[ray_color]["params_bg"],
                alpha=0.4,
            ),
            fontsize=10,
            fontfamily="monospace",
        )

        # Dynamic round-trip counter
        round_text = ax.text(
            0.5,
            0.98,
            f"Round trips: 0/{N_round_trips}",
            transform=ax.transAxes,
            va="top",
            ha="center",
            bbox=dict(
                boxstyle="round",
                facecolor=self.color_dict[ray_color]["timer_bg"],
                edgecolor="grey",
                alpha=0.7,
            ),
            fontsize=12,
            fontfamily="monospace",
        )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(f"Ray Tracing Animation in {title_str} Cavity", pad=10)

        def animate(frame):
            nonlocal arrow_patch
            if frame >= len(all_points):
                return (ray_line, round_text)

            trail = all_points[: frame + 1]
            trail_x = [p[0] for p in trail]
            trail_y = [p[1] for p in trail]
            ray_line.set_data(trail_x, trail_y)

            s_idx = seg_ids[frame]
            completed = s_idx // 2
            if (s_idx % 2 == 1) and (pt_idx_in_seg[frame] == npps - 1):
                completed += 1
            round_text.set_text(f"Round trips: {completed}/{N_round_trips}")

            if arrow_patch is not None:
                arrow_patch.remove()
                arrow_patch = None

            if frame > 0:
                x_prev, y_prev = all_points[frame - 1]
                x_cur, y_cur = all_points[frame]
                dx, dy = x_cur - x_prev, y_cur - y_prev
                if dx * dx + dy * dy > 1e-9:
                    start_x = x_cur - 0.25 * dx
                    start_y = y_cur - 0.25 * dy
                    arrow_patch = FancyArrowPatch(
                        (start_x, start_y),
                        (x_cur, y_cur),
                        arrowstyle="-|>",
                        mutation_scale=16,
                        color=ray_color,
                        linewidth=2,
                        zorder=6,
                        alpha=0.95,
                    )
                    ax.add_patch(arrow_patch)

            return (ray_line, round_text)

        # Animation
        anim = FuncAnimation(
            fig,
            animate,
            frames=len(all_points),
            interval=int(1000 / fps),
            blit=False,
            repeat=False,
        )

        if save_animation:
            self._save_animation(anim, filename, fps, ray_color, anim_format)
            plt.close(fig)
        else:
            plt.show()

        return fig, ax, anim

    def _save_animation(self, anim, filename, fps, ray_color, anim_format="gif"):
        """
        Save animation to file (.gif or .mp4).

        Parameters
        ----------
        anim : matplotlib.animation.FuncAnimation
            Animation object to save
        filename : str or None
            Base filename (extension appended automatically; auto-generated if None)
        fps : int
            Frames per second
        ray_color : str
            Color of the ray (for auto-generated filename metadata)
        anim_format : str
            'gif' or 'mp4'
        """
        anim_format = anim_format.lower().lstrip(".")

        if filename is None:
            config_prefix = self.cavity_config.lower().replace("-", "_")
            sym_prefix = "symmetric" if self.is_symmetric else "asymmetric"
            metadata = f"R1_{self.R1}_R2_{self.R2}_L_{self.L}_rc_{ray_color}"
            filename = f"{sym_prefix}_{config_prefix}_cavity_{metadata}_{timestamp}"

        base = Path(filename).stem
        filepath = ANIMATION_DIR / f"{base}.{anim_format}"

        try:
            print(f"Saving animation to: {filepath}")
            if anim_format == "mp4":
                writer = FFMpegWriter(
                    fps=fps,
                    codec="libx264",
                    bitrate=8000,
                    extra_args=[
                        "-pix_fmt",
                        "yuv420p",
                        "-profile:v",
                        "high",
                        "-level",
                        "4.1",
                        "-movflags",
                        "+faststart",
                    ],
                )
                anim.save(filepath, writer=writer)
            else:
                # GIF: default ffmpeg writer
                anim.save(filepath, writer="ffmpeg", fps=fps, dpi=120)
            print("Animation saved successfully!")
        except Exception as e:
            print(f"Error saving animation: {e}")

    # -------------------------------------------------------------------------
    # Interactive CLI
    # -------------------------------------------------------------------------

    def run_interactive(self):
        """
        Unified interactive CLI for cavity ray tracing.

        Asks the user whether they want a static plot or an animation, then
        collects the appropriate parameters.  The user may also choose to save
        the figure, the animation, or both at the same time.  A single base
        filename is used; the correct extension (.png / .gif / .mp4) is
        appended automatically based on the output type.

        Returns
        -------
        For static plot  : (fig, ax)
        For animation    : (fig, ax, anim)
        """
        print("\n" + "=" * 60)
        print("OPTICAL CAVITY RAY TRACING - INTERACTIVE MODE")
        print("=" * 60)
        print("\nWhat would you like to see?")
        print("  [1] Static plot")
        print("  [2] Animation")
        mode_choice = input("Enter choice (1/2) [1]: ").strip()
        show_animation = mode_choice == "2"

        defaults = {
            "N_round_trips": 25 if show_animation else 50,
            "y0_initial": 15.0,
            "theta0_initial_deg": 0.0,
            "arc_angle": 25,
            "ray_color": "red",
            "fps": 30,
            "points_per_segment": 5,
        }

        use_defaults = (
            input("\nUse default ray parameters? (y/n) [y]: ").strip().lower()
        )
        use_defaults = use_defaults != "n"

        if use_defaults:
            params = defaults.copy()
        else:
            params = defaults.copy()
            try:
                params["N_round_trips"] = int(
                    input(
                        f"Enter number of round trips [{defaults['N_round_trips']}]: "
                    )
                    or defaults["N_round_trips"]
                )
                params["y0_initial"] = float(
                    input(f"Enter initial ray height [{defaults['y0_initial']}]: ")
                    or defaults["y0_initial"]
                )
                params["theta0_initial_deg"] = float(
                    input(
                        f"Enter initial angle in degrees [{defaults['theta0_initial_deg']}]: "
                    )
                    or defaults["theta0_initial_deg"]
                )

                color_choice = (
                    input("Enter ray color (red/blue/green/orange/purple) [red]: ")
                    .strip()
                    .lower()
                )
                params["ray_color"] = (
                    color_choice
                    if color_choice in ["red", "blue", "green", "orange", "purple"]
                    else defaults["ray_color"]
                )

                params["arc_angle"] = float(
                    input(
                        f"Enter arc angle for mirror display [{defaults['arc_angle']}]: "
                    )
                    or defaults["arc_angle"]
                )

                if show_animation:
                    params["points_per_segment"] = int(
                        input(
                            f"Enter points per segment [{defaults['points_per_segment']}]: "
                        )
                        or defaults["points_per_segment"]
                    )
                    params["fps"] = int(
                        input(f"Enter FPS [{defaults['fps']}]: ") or defaults["fps"]
                    )

            except ValueError as e:
                print(f"Invalid input: {e}")
                print("Using default parameters.")
                params = defaults.copy()

        print("\n--- Save Options ---")
        save_figure = (
            input("Save static figure as PNG? (y/n) [n]: ").strip().lower() == "y"
        )
        save_animation = False
        anim_format = "gif"
        show_animation_also = False

        if show_animation:
            save_animation = input("Save animation? (y/n) [n]: ").strip().lower() == "y"
            if save_animation:
                fmt_choice = input("Animation format (gif/mp4) [gif]: ").strip().lower()
                anim_format = fmt_choice if fmt_choice in ("gif", "mp4") else "gif"
        else:
            save_animation_also = (
                input("Also save an animation of this configuration? (y/n) [n]: ")
                .strip()
                .lower()
                == "y"
            )
            if save_animation_also:
                show_animation_also = True
                save_animation = True
                fmt_choice = input("Animation format (gif/mp4) [gif]: ").strip().lower()
                anim_format = fmt_choice if fmt_choice in ("gif", "mp4") else "gif"

        base_filename = None
        if save_figure or save_animation:
            input_filename = input(
                "\nEnter base filename (without extension, or press Enter for auto): "
            ).strip()
            if input_filename:
                base_filename = Path(input_filename).stem

        self.print_info(
            y0_initial=params["y0_initial"],
            theta0_initial_deg=params["theta0_initial_deg"],
            N_round_trips=params["N_round_trips"],
        )

        ray_color = params["ray_color"]
        y0 = params["y0_initial"]
        theta0 = params["theta0_initial_deg"]
        N = params["N_round_trips"]
        arc_angle = params["arc_angle"]

        try:
            if show_animation:
                print(f"\nStarting animation with {N} round trips...")
                if not save_animation:
                    print("Close the plot window to continue.")
                fig, ax, anim = self.animate_cavity(
                    y0_initial=y0,
                    theta0_initial_deg=theta0,
                    N_round_trips=N,
                    arc_angle=arc_angle,
                    fps=params["fps"],
                    points_per_segment=params["points_per_segment"],
                    ray_color=ray_color,
                    save_animation=save_animation,
                    filename=base_filename,
                    anim_format=anim_format,
                )
                if save_figure:
                    static_fig, static_ax = self.visualize_single_ray(
                        y0_initial=y0,
                        theta0_initial_deg=theta0,
                        N_round_trips=N,
                        arc_angle=arc_angle,
                        ray_color=ray_color,
                        save_figure=True,
                        filename=base_filename,
                    )
                return fig, ax, anim

            else:
                fig, ax = self.visualize_single_ray(
                    y0_initial=y0,
                    theta0_initial_deg=theta0,
                    N_round_trips=N,
                    arc_angle=arc_angle,
                    ray_color=ray_color,
                    save_figure=save_figure,
                    filename=base_filename,
                )
                if show_animation_also:
                    print(f"\nGenerating animation with {N} round trips...")
                    anim_fig, anim_ax, anim = self.animate_cavity(
                        y0_initial=y0,
                        theta0_initial_deg=theta0,
                        N_round_trips=N,
                        arc_angle=arc_angle,
                        fps=defaults["fps"],
                        points_per_segment=defaults["points_per_segment"],
                        ray_color=ray_color,
                        save_animation=True,
                        filename=base_filename,
                        anim_format=anim_format,
                    )
                return fig, ax

        except Exception as e:
            print(f"\nError during simulation: {e}")
            raise


def main():
    """Main function for interactive cavity ray tracing."""

    print("=" * 60)
    print("OPTICAL CAVITY RAY TRACING")
    print("=" * 60)

    use_default_cavity = (
        input("Use default cavity parameters? (R1=-80, R2=-80, L=70) (y/n) [y]: ")
        .strip()
        .lower()
    )
    use_default_cavity = use_default_cavity != "n"

    if use_default_cavity:
        R1, R2, L = -80.0, -80.0, 70.0
    else:
        try:
            R1 = float(input("Enter R1 (- for concave, + for convex) [-80]: ") or -80.0)
            R2 = float(input("Enter R2 (- for concave, + for convex) [-80]: ") or -80.0)
            L = float(input("Enter cavity length L [70]: ") or 70.0)
        except ValueError:
            print("Invalid input, using defaults.")
            R1, R2, L = -80.0, -80.0, 70.0

    try:
        cavity = CavityRayTracing(R1=R1, R2=R2, L=L)
        return cavity.run_interactive()
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    main()
