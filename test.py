from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import FancyArrowPatch


from ray_tracing import CavityRayTracing, PLOT_DIR, ANIMATION_DIR, timestamp


CAVITY_CASES = {
    "concave_concave": {
        "CC01_symmetric_confocal_boundary": {
            "R1": -80.0,
            "R2": -80.0,
            "L": 80.0,
            "y0_initial": 15.0,
            "theta0_initial_deg": 0.0,
            "N_round_trips": 10,
            "arc_angle": 30.0,
            "ray_color": "red",
        },
        "CC02_symmetric_near_concentric": {
            "R1": -50.0,
            "R2": -50.0,
            "L": 80.0,
            "y0_initial": 12.0,
            "theta0_initial_deg": 1.5,
            "N_round_trips": 40,
            "arc_angle": 40.0,
            "ray_color": "blue",
        },
        "CC03_asymmetric_near_concentric": {
            "R1": -50.0,
            "R2": -80.0,
            "L": 95.0,
            "y0_initial": -12.0,
            "theta0_initial_deg": 1.0,
            "N_round_trips": 32,
            "arc_angle": 30.0,
            "ray_color": "green",
        },
        "CC04_asymmetric_concentric_side": {
            "R1": -85.0,
            "R2": -40.0,
            "L": 90.0,
            "y0_initial": 6.0,
            "theta0_initial_deg": 0.0,
            "N_round_trips": 30,
            "arc_angle": 25.0,
            "ray_color": "purple",
        },
    },
    "concave_convex": {
        "CX01_near_upper_stability_boundary": {
            "R1": -140.0,
            "R2": 70.0,
            "L": 85.0,
            "y0_initial": 5.0,
            "theta0_initial_deg": 0.0,
            "N_round_trips": 30,
            "arc_angle": 20.0,
            "ray_color": "orange",
        },
        "CX02_focused_concave_convex": {
            "R1": -100.0,
            "R2": 50.0,
            "L": 90.0,
            "y0_initial": 5.5,
            "theta0_initial_deg": 0.0,
            "N_round_trips": 35,
            "arc_angle": 25.0,
            "ray_color": "red",
        },
        "CX03_focused_convex_concave": {
            "R1": 50.0,
            "R2": -100.0,
            "L": 95.0,
            "y0_initial": -6.5,
            "theta0_initial_deg": 0.0,
            "N_round_trips": 40,
            "arc_angle": 35.0,
            "ray_color": "blue",
        },
        "CX04_low_product_convex_concave": {
            "R1": 60.0,
            "R2": -120.0,
            "L": 115.0,
            "y0_initial": 6.0,
            "theta0_initial_deg": 0.0,
            "N_round_trips": 30,
            "arc_angle": 30.0,
            "ray_color": "green",
        },
    },
}


def _draw_case(ax, case_name, case_parameters):
    """Draw one configured cavity case on an existing Matplotlib axis."""

    cavity = CavityRayTracing(
        R1=case_parameters["R1"],
        R2=case_parameters["R2"],
        L=case_parameters["L"],
    )

    ray_path, _ = cavity.trace_single_ray(
        y0_initial=case_parameters["y0_initial"],
        theta0_initial_deg=case_parameters["theta0_initial_deg"],
        N_round_trips=case_parameters["N_round_trips"],
    )
    left_mirror, right_mirror = cavity._get_mirror_arcs(case_parameters["arc_angle"])

    ray_color = case_parameters["ray_color"]
    ax.set_facecolor(cavity.color_dict[ray_color]["ax_face"])
    ax.add_patch(left_mirror)
    ax.add_patch(right_mirror)

    ax.plot(
        [-cavity.L / 2, cavity.L / 2],
        [0, 0],
        color="grey",
        lw=1,
        ls="--",
        zorder=1,
        alpha=0.6,
    )

    path_x, path_y = zip(*ray_path)
    ax.plot(
        path_x,
        path_y,
        "-o",
        color=ray_color,
        markersize=2.5,
        linewidth=1.5,
        zorder=3,
    )

    params = cavity.get_cavity_parameters()
    status = "STABLE" if params.is_stable else "UNSTABLE"
    if params.is_confocal:
        title_kind = "Confocal"
    elif params.is_symmetric:
        title_kind = f"Symmetric {params.mirror_description}"
    else:
        title_kind = f"Asymmetric {params.mirror_description}"

    ax.set_title(f"{title_kind} {status}", fontsize=12, pad=8)
    ax.text(
        0.5,
        0.98,
        (
            f"$R_1$ = {params.R1:.1f} cm, "
            f"$R_2$ = {params.R2:.1f} cm, "
            f"$L$ = {params.L:.1f} cm\n"
            f"$g_1$ = {params.g1:.3f}, "
            f"$g_2$ = {params.g2:.3f}, "
            f"$g_1 \\times g_2$ = {params.stability_product:.3f}\n"
            f"$y_0$ = {case_parameters['y0_initial']:.1f} cm, "
            f"$\\theta_0$ = {case_parameters['theta0_initial_deg']:.1f}°, "
            f"Trips = {case_parameters['N_round_trips']}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="center",
        fontsize=8,
        family="monospace",
        bbox={
            "boxstyle": "round",
            "facecolor": cavity.color_dict[ray_color]["params_bg"],
            "alpha": 0.45,
        },
        zorder=6,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return cavity


def plot_cavity_cases(
    cases=None,
    save_figures=False,
    output_dir=PLOT_DIR,
    show=True,
    choice=None,
):
    """Plot concave-concave and/or concave-convex cases in separate 2x2 figures.

    Parameters
    ----------
    cases : dict or None
        Dictionary with the same structure as ``CAVITY_CASES``. If omitted,
        the four built-in cases for each cavity category are used.
    save_figures : bool
        Save composite figures as PNG files when ``True``.
    output_dir : str or pathlib.Path
        Directory used when ``save_figures`` is enabled.
    show : bool
        Call ``plt.show()`` after creating the figures.
    choice : str or None
        Which cases to plot. Options: 'concave_concave', 'concave_convex', 'both', or None.
        If None, the user will be prompted to choose interactively.

    Returns
    -------
    tuple
        ``(concave_concave_figure, concave_convex_figure)``. If a case category is
        not selected, its respective figure in the returned tuple will be ``None``.
    """

    if choice is None:
        print("Which cavity cases would you like to plot?")
        print("1. Concave-Concave Cavities")
        print("2. Concave-Convex Cavities")
        print("3. Both")
        user_input = input("Enter choice (1, 2, or 3) [3]: ").strip()
        if user_input == "1":
            choice = "concave_concave"
        elif user_input == "2":
            choice = "concave_convex"
        else:
            choice = "both"

    selected_cases = CAVITY_CASES if cases is None else cases
    figures = {}

    all_specs = {
        "concave_concave": (
            "Concave-Concave Cavities",
            "concave_concave_2x2.png",
        ),
        "concave_convex": (
            "Concave-Convex Cavities",
            "concave_convex_2x2.png",
        ),
    }

    if choice == "concave_concave":
        figure_specs = {"concave_concave": all_specs["concave_concave"]}
    elif choice == "concave_convex":
        figure_specs = {"concave_convex": all_specs["concave_convex"]}
    else:
        figure_specs = all_specs

    for category, (figure_title, filename) in figure_specs.items():
        category_cases = selected_cases[category]
        if len(category_cases) != 4:
            raise ValueError(
                f"{category} must contain exactly four cases for a 2x2 figure"
            )

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.subplots_adjust(
            left=0.03,
            right=0.97,
            top=0.92,
            bottom=0.03,
            hspace=0.22,
            wspace=0.08,
        )
        fig.patch.set_facecolor("black")
        fig.suptitle(figure_title, fontsize=18, fontweight="bold")

        for ax, (case_name, case_parameters) in zip(axes.flat, category_cases.items()):
            _draw_case(ax, case_name, case_parameters)

        figures[category] = fig

        if save_figures:
            save_directory = Path(output_dir)
            save_directory.mkdir(parents=True, exist_ok=True)

            choice_prompt = f"For {figure_title}, save as default '{filename}' (d) or enter custom filename (c)? [d]: "
            filename_choice = input(choice_prompt).strip().lower()

            final_filename = filename
            if filename_choice == "c":
                custom_name = input(
                    "Enter custom filename (e.g. custom_name.png): "
                ).strip()
                if custom_name:
                    if not custom_name.endswith(".png"):
                        custom_name += ".png"
                    final_filename = custom_name
            else:
                base_path = save_directory / final_filename
                if base_path.exists():
                    stem = base_path.stem
                    ext = base_path.suffix
                    counter = 1
                    while True:
                        new_filename = f"{stem}_{counter}{ext}"
                        new_path = save_directory / new_filename
                        if not new_path.exists():
                            final_filename = new_filename
                            break
                        counter += 1

            save_path = save_directory / final_filename
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved successfully to: {save_path}")

    if show and figures:
        plt.show()

    return figures.get("concave_concave"), figures.get("concave_convex")


def animate_cavity_cases(
    cases=None,
    animate=False,
    save_figure=False,
    save_animation=False,
    filename=None,
    output_dir=PLOT_DIR,
    anim_dir=ANIMATION_DIR,
    choice=None,
    fps=30,
    points_per_segment=5,
    anim_format="gif",
):
    """Animate or plot cavity ray tracing for 4 cases simultaneously in a 2×2 grid.

    The caller (or ``__main__``) decides the display mode via ``animate``.
    Save options are collected here via a concise interactive menu.

    Parameters
    ----------
    cases : dict or None
        Dictionary with the same structure as ``CAVITY_CASES``.  If omitted,
        the built-in cases are used.
    animate : bool
        ``True``  → build and display the animation (default ``False``).
        ``False`` → show the static 2×2 figure.
    save_figure : bool
        Save the static composite figure as a PNG file (default ``False``).
    save_animation : bool
        Save the animation to disk (default ``False``).
    filename : str or None
        Base filename **without** extension.  The correct extension
        (``.png`` / ``.gif`` / ``.mp4``) is appended automatically.
        A timestamped default name is used when ``None``.
    output_dir : str or pathlib.Path
        Directory for saved PNG figures.
    anim_dir : str or pathlib.Path
        Directory for saved animations.
    choice : str or None
        Which category to render: ``'concave_concave'`` or
        ``'concave_convex'``.  Defaults to ``'concave_concave'`` when
        ``None`` and the user presses Enter.
    fps : int
        Frames per second for the animation.
    points_per_segment : int
        Interpolation points per ray segment (controls animation smoothness).
    anim_format : str
        ``'gif'`` or ``'mp4'`` (default ``'gif'``).

    Returns
    -------
    results : dict
        Keys are category names (``'concave_concave'``, ``'concave_convex'``).
        Each value is a dict with keys ``'fig'`` and, when in animation mode,
        ``'anim'``.
    """

    _animate = animate
    _save_figure = save_figure
    _save_animation = save_animation
    _anim_format = anim_format

    print("\n--- Save Options ---")

    if _animate:
        print("  [1] Don't save anything  (default)")
        print("  [2] Save animation only")
        print("  [3] Save static figure only")
        print("  [4] Save both animation and static figure")
        save_choice = input("Enter choice (1/2/3/4) [1]: ").strip()
        if save_choice == "2":
            _save_animation = True
        elif save_choice == "3":
            _save_figure = True
        elif save_choice == "4":
            _save_animation = True
            _save_figure = True

        if _save_animation:
            fmt_input = input("Animation format (gif/mp4) [gif]: ").strip().lower()
            _anim_format = fmt_input if fmt_input in ("gif", "mp4") else "gif"
    else:
        if not _save_figure:
            _save_figure = (
                input("Save static figure as PNG? (y/n) [n]: ").strip().lower() == "y"
            )

    _filename = filename
    if (_save_figure or _save_animation) and _filename is None:
        input_name = input(
            "\nEnter base filename (without extension, or press Enter for auto): "
        ).strip()
        if input_name:
            _filename = Path(input_name).stem

    _auto_stem = f"cavity_4panel_{timestamp}"

    if choice is None:
        print("\nWhich cavity cases would you like to render?")
        print("  [1] Concave-Concave Cavities  (default)")
        print("  [2] Concave-Convex Cavities")
        cat_input = input("Enter choice (1/2) [1]: ").strip()
        if cat_input == "2":
            choice = "concave_convex"
        else:
            choice = "concave_concave"

    selected_cases = CAVITY_CASES if cases is None else cases

    all_specs = {
        "concave_concave": "Concave-Concave Cavities",
        "concave_convex": "Concave-Convex Cavities",
    }

    if choice == "concave_concave":
        figure_specs = {"concave_concave": all_specs["concave_concave"]}
    else:
        figure_specs = {"concave_convex": all_specs["concave_convex"]}

    results = {}

    for category, figure_title in figure_specs.items():
        category_cases = selected_cases[category]
        if len(category_cases) != 4:
            raise ValueError(
                f"{category} must contain exactly four cases for a 2×2 layout"
            )

        if not _animate or _save_figure:
            fig_static, axes_static = plt.subplots(2, 2, figsize=(14, 10))
            fig_static.subplots_adjust(
                left=0.03, right=0.97, top=0.92, bottom=0.03, hspace=0.18, wspace=0.08
            )
            fig_static.patch.set_facecolor("black")
            fig_static.suptitle(figure_title, fontsize=18, fontweight="bold")

            for ax, (case_name, case_params) in zip(
                axes_static.flat, category_cases.items()
            ):
                _draw_case(ax, case_name, case_params)

            if _save_figure:
                _png_dir = Path(output_dir)
                _png_dir.mkdir(parents=True, exist_ok=True)
                stem = _filename if _filename else f"{_auto_stem}_{category}"
                candidate = _png_dir / f"{stem}.png"
                if candidate.exists():
                    counter = 1
                    while True:
                        candidate = _png_dir / f"{stem}_{counter}.png"
                        if not candidate.exists():
                            break
                        counter += 1
                fig_static.savefig(candidate, dpi=300, bbox_inches="tight")
                print(f"Figure saved to: {candidate}")

            if not _animate:
                plt.show()

            results[category] = {"fig": fig_static}

        if _animate or _save_animation:
            anim_fig, anim_axes = plt.subplots(2, 2, figsize=(14, 10))
            anim_fig.subplots_adjust(
                left=0.03, right=0.97, top=0.92, bottom=0.05, hspace=0.18, wspace=0.08
            )
            anim_fig.patch.set_facecolor("black")
            anim_fig.suptitle(f"{figure_title}", fontsize=18, fontweight="bold")

            ray_lines = []
            arrow_patches = [None, None, None, None]
            round_texts = []
            all_points_list = []
            seg_ids_list = []
            pt_idx_list = []
            npps_list = []
            n_trips_list = []

            for idx, (ax, (case_name, case_params)) in enumerate(
                zip(anim_axes.flat, category_cases.items())
            ):
                cavity = CavityRayTracing(
                    R1=case_params["R1"],
                    R2=case_params["R2"],
                    L=case_params["L"],
                )

                ray_color = case_params["ray_color"]
                N = case_params["N_round_trips"]
                n_trips_list.append(N)

                ray_segments, _ = cavity.trace_ray(
                    y0_initial=case_params["y0_initial"],
                    theta0_initial_deg=case_params["theta0_initial_deg"],
                    N_round_trips=N,
                )
                all_pts, seg_ids, pt_idx, npps = cavity._interpolate_segments(
                    ray_segments, points_per_segment
                )
                all_points_list.append(all_pts)
                seg_ids_list.append(seg_ids)
                pt_idx_list.append(pt_idx)
                npps_list.append(npps)

                ax.set_facecolor(cavity.color_dict[ray_color]["ax_face"])
                left_mirror, right_mirror = cavity._get_mirror_arcs(
                    case_params["arc_angle"]
                )
                ax.add_patch(left_mirror)
                ax.add_patch(right_mirror)
                ax.plot(
                    [-cavity.L / 2, cavity.L / 2],
                    [0, 0],
                    color="grey",
                    lw=1,
                    ls="--",
                    zorder=1,
                    alpha=0.6,
                )

                params_obj = cavity.get_cavity_parameters()
                status = "STABLE" if params_obj.is_stable else "UNSTABLE"
                if params_obj.is_confocal:
                    title_kind = "Confocal"
                elif params_obj.is_symmetric:
                    title_kind = f"Symmetric {params_obj.mirror_description}"
                else:
                    title_kind = f"Asymmetric {params_obj.mirror_description}"

                ax.set_title(f"{title_kind} {status}", fontsize=14, pad=8)
                ax.text(
                    0.5,
                    0.98,
                    (
                        f"$R_1$ = {params_obj.R1:.1f} cm, "
                        f"$R_2$ = {params_obj.R2:.1f} cm, "
                        f"$L$ = {params_obj.L:.1f} cm\n"
                        f"$g_1$ = {params_obj.g1:.3f}, "
                        f"$g_2$ = {params_obj.g2:.3f}, "
                        f"$g_1 \\times g_2$ = {params_obj.stability_product:.3f}\n"
                        f"$y_0$ = {case_params['y0_initial']:.1f} cm, "
                        f"$\\theta_0$ = {case_params['theta0_initial_deg']:.1f}°, "
                        f"Trips = {N}"
                    ),
                    transform=ax.transAxes,
                    va="top",
                    ha="center",
                    fontsize=8,
                    family="monospace",
                    bbox={
                        "boxstyle": "round",
                        "facecolor": cavity.color_dict[ray_color]["params_bg"],
                        "alpha": 0.45,
                    },
                    zorder=6,
                )

                rt_text = ax.text(
                    0.5,
                    0.03,
                    f"Round trips: 0/{N}",
                    transform=ax.transAxes,
                    va="bottom",
                    ha="center",
                    bbox=dict(
                        boxstyle="round",
                        facecolor=cavity.color_dict[ray_color]["timer_bg"],
                        edgecolor="grey",
                        alpha=0.7,
                    ),
                    fontsize=9,
                    fontfamily="monospace",
                    zorder=7,
                )
                round_texts.append(rt_text)

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])

                (line,) = ax.plot(
                    [], [], "-", linewidth=1.8, color=ray_color, zorder=3, alpha=0.9
                )
                ray_lines.append(line)

            total_frames = max(len(pts) for pts in all_points_list)

            def _make_update(n_subplots):
                """Return the FuncAnimation update callable."""

                def animate(frame):
                    artists = []
                    for i in range(n_subplots):
                        pts = all_points_list[i]
                        if frame >= len(pts):
                            artists.append(ray_lines[i])
                            artists.append(round_texts[i])
                            continue

                        trail = pts[: frame + 1]
                        ray_lines[i].set_data(
                            [p[0] for p in trail],
                            [p[1] for p in trail],
                        )

                        s_idx = seg_ids_list[i][frame]
                        completed = s_idx // 2
                        if s_idx % 2 == 1 and pt_idx_list[i][frame] == npps_list[i] - 1:
                            completed += 1
                        round_texts[i].set_text(
                            f"Round trips: {completed}/{n_trips_list[i]}"
                        )

                        if arrow_patches[i] is not None:
                            try:
                                arrow_patches[i].remove()  # type: ignore
                            except Exception:
                                pass
                            arrow_patches[i] = None

                        if frame > 0 and frame < len(pts):
                            x_prev, y_prev = pts[frame - 1]
                            x_cur, y_cur = pts[frame]
                            dx, dy = x_cur - x_prev, y_cur - y_prev
                            if dx * dx + dy * dy > 1e-9:
                                start_x = x_cur - 0.25 * dx
                                start_y = y_cur - 0.25 * dy
                                ax_i = anim_axes.flat[i]
                                ray_color_i = list(category_cases.values())[i][
                                    "ray_color"
                                ]
                                arrow = FancyArrowPatch(
                                    (start_x, start_y),
                                    (x_cur, y_cur),
                                    arrowstyle="-|>",
                                    mutation_scale=12,
                                    color=ray_color_i,
                                    linewidth=1.5,
                                    zorder=6,
                                    alpha=0.95,
                                )
                                ax_i.add_patch(arrow)
                                arrow_patches[i] = arrow  # type: ignore

                        artists.append(ray_lines[i])
                        artists.append(round_texts[i])

                    return artists

                return animate

            anim = FuncAnimation(
                anim_fig,
                _make_update(len(list(category_cases.keys()))),
                frames=total_frames,
                interval=int(1000 / fps),
                blit=False,
                repeat=False,
            )

            if _save_animation:
                _anim_dir = Path(anim_dir)
                _anim_dir.mkdir(parents=True, exist_ok=True)
                stem = _filename if _filename else f"{_auto_stem}_{category}"
                anim_path = _anim_dir / f"{stem}.{_anim_format}"
                try:
                    print(f"Saving animation to: {anim_path}")
                    if _anim_format == "mp4":
                        writer = FFMpegWriter(
                            fps=fps,
                            codec="libx264",
                            extra_args=[
                                "-preset",
                                "slow",
                                "-crf",
                                "18",
                                "-pix_fmt",
                                "yuv420p",
                                "-level",
                                "5.1",
                                "-movflags",
                                "+faststart",
                            ],
                        )
                        anim.save(anim_path, writer=writer)
                    else:
                        anim.save(anim_path, writer="ffmpeg", fps=fps, dpi=120)
                    print("Animation saved successfully!")
                except Exception as exc:
                    print(f"Error saving animation: {exc}")

                plt.close(anim_fig)
            else:
                plt.show()

            if category in results:
                results[category]["anim"] = anim
                results[category]["anim_fig"] = anim_fig
            else:
                results[category] = {"anim_fig": anim_fig, "anim": anim}

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("OPTICAL CAVITY RAY TRACING – MULTI-PANEL")
    print("=" * 60)
    print("\nWhat would you like to see?")
    print("  [1] Static figures  (default)")
    print("  [2] Animation")
    main_choice = input("Enter choice (1/2) [1]: ").strip()

    if main_choice == "2":
        animate_cavity_cases(animate=True)
    else:
        animate_cavity_cases(animate=False)
