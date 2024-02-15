from pathlib import Path
import dolfin

from load_geometry import load_ep_mesh, load_mechanics_mesh


def main():
    datadir = Path("data")
    geo_ep = load_ep_mesh(datadir)
    geo_mech = load_mechanics_mesh(datadir)

    V_ep = dolfin.FunctionSpace(geo_ep.mesh, "DG", 0)
    V_mech = dolfin.FunctionSpace(geo_mech.mesh, "DG", 0)
    Ta_ep = dolfin.Function(V_ep)
    Ta_mech = dolfin.Function(V_mech)

    outdir = Path("out_Ta")
    outdir.mkdir(exist_ok=True)
    i = 0
    with dolfin.XDMFFile("results/state_ORdmm_Land_Ta.xdmf") as xdmf:
        while True:
            try:
                xdmf.read_checkpoint(Ta_ep, "Ta_dg", i)
            except RuntimeError:
                break
            else:
                i += 1
                Ta_mech.interpolate(Ta_ep)
                (outdir / f"out_{i}.txt").write_text(
                    "\n".join(
                        map(
                            lambda x: " ".join(map(str, x)),
                            tuple(enumerate(2 * Ta_mech.vector().get_local())),
                        )
                    )
                )


if __name__ == "__main__":
    main()
