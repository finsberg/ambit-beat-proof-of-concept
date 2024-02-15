from collections import defaultdict
from pathlib import Path
import cardiac_geometries
import numpy as np
import matplotlib.pyplot as plt
import dolfin
import ufl_legacy as ufl

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

import beat
import gotranx

import load_geometry


def get_data(datadir="data_endocardial_stimulation"):
    datadir = Path(datadir)
    msh_file = datadir / "biv_ellipsoid.msh"
    if not msh_file.is_file():
        cardiac_geometries.create_biv_ellipsoid(
            datadir,
            char_length=0.2,  # Reduce this value to get a finer mesh
            center_lv_y=0.2,
            center_lv_z=0.0,
            a_endo_lv=5.0,
            b_endo_lv=2.2,
            c_endo_lv=2.2,
            a_epi_lv=6.0,
            b_epi_lv=3.0,
            c_epi_lv=3.0,
            center_rv_y=1.0,
            center_rv_z=0.0,
            a_endo_rv=6.0,
            b_endo_rv=2.5,
            c_endo_rv=2.7,
            a_epi_rv=8.0,
            b_epi_rv=5.5,
            c_epi_rv=4.0,
            create_fibers=True,
        )

    return cardiac_geometries.geometry.Geometry.from_folder(datadir)


def define_stimulus(mesh, chi, C_m, time, ffun, markers):
    duration = 2.0  # ms
    # A = 5  # mu A/cm^3
    A = 0.07

    # factor = (1 / 50) * 1.0 / (chi * C_m)  # NB: cbcbeat convention
    factor = 1.0 / (chi * C_m)  # NB: cbcbeat convention
    amplitude = factor * A  # mV/ms

    I_s = dolfin.Expression(
        "time >= start ? (time <= (duration + start) ? amplitude : 0.0) : 0.0",
        time=time,
        start=0.0,
        duration=duration,
        amplitude=amplitude,
        degree=0,
    )

    subdomain_data = dolfin.MeshFunction("size_t", mesh, 2)
    subdomain_data.set_all(0)
    marker = 1
    subdomain_data.array()[ffun.array() == markers["ENDO_LV"][0]] = 1
    subdomain_data.array()[ffun.array() == markers["ENDO_RV"][0]] = 1

    ds = dolfin.Measure("ds", domain=mesh, subdomain_data=subdomain_data)(marker)
    return beat.base_model.Stimulus(dz=ds, expr=I_s)


def define_conductivity_tensor(chi, C_m, f0, s0, n0):
    # Conductivities as defined by page 4339 of Niederer benchmark
    # sigma_il = 0.17  # mS / mm
    # sigma_it = 0.019  # mS / mm
    # sigma_el = 0.62  # mS / mm
    # sigma_et = 0.24  # mS / mm

    sigma_il = 0.34
    sigma_it = 0.060
    sigma_el = 0.12
    sigma_et = 0.080

    # Compute monodomain approximation by taking harmonic mean in each
    # direction of intracellular and extracellular part
    def harmonic_mean(a, b):
        return a * b / (a + b)

    sigma_l = harmonic_mean(sigma_il, sigma_el)
    sigma_t = harmonic_mean(sigma_it, sigma_et)

    # Scale conducitivites by 1/(C_m * chi)
    s_l = sigma_l / (C_m * chi)  # mm^2 / ms
    s_t = sigma_t / (C_m * chi)  # mm^2 / ms

    # Define conductivity tensor
    A = dolfin.as_matrix(
        [
            [f0[0], s0[0], n0[0]],
            [f0[1], s0[1], n0[1]],
            [f0[2], s0[2], n0[2]],
        ],
    )

    M_star = ufl.diag(dolfin.as_vector([s_l, s_t, s_t]))
    M = A * M_star * A.T

    return M


def load_timesteps_from_xdmf(xdmffile):
    import xml.etree.ElementTree as ET

    times = {}
    i = 0
    tree = ET.parse(xdmffile)
    for elem in tree.iter():
        if elem.tag == "Time":
            times[i] = float(elem.get("Value"))
            i += 1

    return times


def load_from_file(heart_mesh, xdmffile, key="v", stop_index=None):
    V = dolfin.FunctionSpace(heart_mesh, "Lagrange", 1)
    v = dolfin.Function(V)

    timesteps = load_timesteps_from_xdmf(xdmffile)
    with dolfin.XDMFFile(Path(xdmffile).as_posix()) as f:
        for i, t in tqdm(timesteps.items()):
            f.read_checkpoint(v, key, i)
            yield v.copy(deepcopy=True), t


def main():
    datadir = Path("results")

    data = load_geometry.main()
    data.mesh.coordinates()

    V = dolfin.FunctionSpace(data.mesh, "Lagrange", 1)

    markers = dolfin.Function(V)
    arr = beat.utils.expand_layer(
        markers=markers,
        mfun=data.ffun,
        endo_markers=[data.markers["ENDO_LV"][0], data.markers["ENDO_RV"][0]],
        epi_markers=[data.markers["EPI"][0]],
        endo_marker=1,
        epi_marker=2,
        endo_size=0.3,
        epi_size=0.3,
    )

    markers.vector().set_local(arr)

    with dolfin.XDMFFile((datadir / "markers.xdmf").as_posix()) as xdmf:
        xdmf.write(markers)

    ode = gotranx.load_ode("ORdmm_Land.ode")
    code = gotranx.cli.gotran2py.get_code(
        ode, scheme=[gotranx.schemes.Scheme.forward_generalized_rush_larsen]
    )
    model = {}
    exec(code, model)
    data = get_data(datadir=datadir)

    init_states = {
        0: model["init_state_values"](),
        1: model["init_state_values"](),
        2: model["init_state_values"](),
    }
    # endo = 0, epi = 1, M = 2
    parameters = {
        0: model["init_parameter_values"](amp=0.0, celltype=2),
        1: model["init_parameter_values"](amp=0.0, celltype=0),
        2: model["init_parameter_values"](amp=0.0, celltype=1),
    }
    fun = {
        0: model["forward_generalized_rush_larsen"],
        1: model["forward_generalized_rush_larsen"],
        2: model["forward_generalized_rush_larsen"],
    }
    v_index = {
        0: model["state_index"]("v"),
        1: model["state_index"]("v"),
        2: model["state_index"]("v"),
    }

    # TODO: Need to figure out these values
    # Surface to volume ratio
    # chi = 140.0  # mm^{-1}
    # # Membrane capacitance
    # C_m = 0.01  # mu F / mm^2

    # Surface to volume ratio
    chi = 1 / 1400  # mm^{-1}
    # Membrane capacitance
    C_m = 1  # mu F / mm^2

    time = dolfin.Constant(0.0)
    I_s = define_stimulus(
        mesh=data.mesh,
        chi=chi,
        C_m=C_m,
        time=time,
        ffun=data.ffun,
        markers=data.markers,
    )

    M = define_conductivity_tensor(chi, C_m, f0=data.f0, s0=data.s0, n0=data.n0)

    params = {"preconditioner": "sor", "use_custom_preconditioner": False}
    pde = beat.MonodomainModel(time=time, mesh=data.mesh, M=M, I_s=I_s, params=params)

    ode = beat.odesolver.DolfinMultiODESolver(
        pde.state,
        markers=markers,
        num_states={i: len(s) for i, s in init_states.items()},
        fun=fun,
        init_states=init_states,
        parameters=parameters,
        # monitor=monitor,
        v_index=v_index,
    )

    T = 1000

    t = 0.0
    dt = 0.05
    solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode)

    Ta = dolfin.Function(pde.V)
    Ta_index = model["monitor_index"]("Ta")
    V_dg0 = dolfin.FunctionSpace(data.mesh, "DG", 0)
    Ta_dg = dolfin.Function(V_dg0)

    Taname = (datadir / "Ta.xdmf").as_posix()
    fname = (datadir / "V.xdmf").as_posix()
    i = 0
    while t < T + 1e-12:
        # Make sure to save at the same time steps that is used by Ambit
        if i % 40 == 0:
            v = solver.pde.state.vector().get_local()
            print(f"Solve for {t=:.2f}, {v.max() =}, {v.min() = }")
            with dolfin.XDMFFile(dolfin.MPI.comm_world, fname) as xdmf:
                xdmf.write_checkpoint(
                    solver.pde.state,
                    "V",
                    float(t),
                    dolfin.XDMFFile.Encoding.HDF5,
                    True,
                )

            arr = Ta.vector().get_local().copy()

            for marker in ode._marker_values:
                monitor_values = model["monitor"](
                    t, solver.ode.values(marker=marker), solver.ode.parameters[marker]
                )
                arr[ode._inds[marker]] = monitor_values[Ta_index]
            Ta.vector().set_local(arr)
            with dolfin.XDMFFile(dolfin.MPI.comm_world, Taname) as xdmf:
                xdmf.write_checkpoint(
                    Ta,
                    "Ta",
                    float(t),
                    dolfin.XDMFFile.Encoding.HDF5,
                    True,
                )
            Ta_dg.interpolate(Ta)
            with dolfin.XDMFFile(dolfin.MPI.comm_world, Taname) as xdmf:
                xdmf.write_checkpoint(
                    Ta_dg,
                    "Ta_dg",
                    float(t),
                    dolfin.XDMFFile.Encoding.HDF5,
                    True,
                )

        solver.step((t, t + dt))
        i += 1
        t += dt


if __name__ == "__main__":
    main()
