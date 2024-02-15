from typing import NamedTuple
from pathlib import Path
import dolfin as df
import ldrb

df.parameters["refinement_algorithm"] = "plaza_with_parent_facets"


class Geometry(NamedTuple):
    mesh: df.Mesh
    ffun: df.MeshFunction
    markers: dict[str, tuple[int, int]]
    f0: df.Function
    s0: df.Function
    n0: df.Function

    def save(self, outdir: str):
        Path(outdir).mkdir(parents=True, exist_ok=True)
        with df.XDMFFile(df.MPI.comm_world, f"{outdir}/mesh.xdmf") as xdmf:
            xdmf.write(self.mesh)
        with df.XDMFFile(df.MPI.comm_world, f"{outdir}/ffun.xdmf") as xdmf:
            xdmf.write(self.ffun)
        filename = f"{outdir}/microstructure.xdmf"
        with df.XDMFFile(df.MPI.comm_world, filename) as xdmf:
            xdmf.write_checkpoint(
                self.f0, "fiber", 0, df.XDMFFile.Encoding.HDF5, append=True
            )
            xdmf.write_checkpoint(
                self.s0, "sheet", 0, df.XDMFFile.Encoding.HDF5, append=True
            )
            xdmf.write_checkpoint(
                self.n0, "sheet-normal", 0, df.XDMFFile.Encoding.HDF5, append=True
            )


def load_mechanics_mesh(datadir: Path):
    mech_outdir = datadir / "mechanics"
    if mech_outdir.is_dir():
        return load_geometry_from_folder(mech_outdir)
    mesh = df.Mesh()
    with df.XDMFFile("input/heart3D_domain.xdmf") as infile:
        infile.read(mesh)
    ffun_val = df.MeshValueCollection("size_t", mesh, 2)
    with df.XDMFFile("input/heart3D_boundary.xdmf") as infile:
        infile.read(ffun_val)
    ffun = df.MeshFunction("size_t", mesh, ffun_val)
    ffun.array()[ffun.array() == max(ffun.array())] = 0

    markers = {
        "ENDO_LV": [1, 2],
        "ENDO_RV": [2, 2],
        "EPI": [3, 2],
        "BASE": [4, 2],
    }
    ldrb_markers = {
        "base": markers["BASE"][0],
        "lv": markers["ENDO_LV"][0],
        "rv": markers["ENDO_RV"][0],
        "epi": markers["EPI"][0],
    }
    f0, s0, n0 = ldrb.dolfin_ldrb(
        mesh=mesh,
        fiber_space="CG_1",
        ffun=ffun,
        markers=ldrb_markers,
        alpha_endo_lv=60,  # Fiber angle on the endocardium
        alpha_endo_rv=80,  # Fiber angle on the endocardium
        alpha_epi_lv=-30,  # Fiber angle on the epicardium
        beta_endo_lv=0,
        beta_epi_lv=0,
    )

    geo = Geometry(mesh, ffun, markers, f0, s0, n0)
    geo.save(mech_outdir)
    return geo


def refine_geometry(geometry: Geometry, refine: int):
    mesh = geometry.mesh
    ffun = geometry.ffun
    markers = geometry.markers
    f0 = geometry.f0
    s0 = geometry.s0
    n0 = geometry.n0

    for _ in range(refine):
        mesh = df.adapt(mesh)
        ffun = df.adapt(ffun, mesh)

    ldrb_markers = {
        "base": markers["BASE"][0],
        "lv": markers["ENDO_LV"][0],
        "rv": markers["ENDO_RV"][0],
        "epi": markers["EPI"][0],
    }
    f0, s0, n0 = ldrb.dolfin_ldrb(
        mesh=mesh,
        fiber_space="CG_1",
        ffun=ffun,
        markers=ldrb_markers,
        alpha_endo_lv=60,  # Fiber angle on the endocardium
        alpha_endo_rv=80,  # Fiber angle on the endocardium
        alpha_epi_lv=-30,  # Fiber angle on the epicardium
        beta_endo_lv=0,
        beta_epi_lv=0,
    )

    return Geometry(mesh, ffun, markers, f0, s0, n0)


def load_geometry_from_folder(folder: str):
    markers = {
        "ENDO_LV": [1, 2],
        "ENDO_RV": [2, 2],
        "EPI": [3, 2],
        "BASE": [4, 2],
    }
    mesh = df.Mesh()
    with df.XDMFFile(f"{folder}/mesh.xdmf") as infile:
        infile.read(mesh)
    ffun = df.MeshFunction("size_t", mesh, 2)
    with df.XDMFFile(f"{folder}/ffun.xdmf") as infile:
        infile.read(ffun)
    V = df.VectorFunctionSpace(mesh, "CG", 1)
    f0 = df.Function(V)
    s0 = df.Function(V)
    n0 = df.Function(V)
    with df.XDMFFile(f"{folder}/microstructure.xdmf") as infile:
        infile.read_checkpoint(f0, "fiber")
        infile.read_checkpoint(s0, "sheet")
        infile.read_checkpoint(n0, "sheet-normal")

    return Geometry(mesh, ffun, markers, f0, s0, n0)


def load_ep_mesh(datadir: Path):
    ep_outdir = datadir / "ep"
    if ep_outdir.is_dir():
        return load_geometry_from_folder(ep_outdir)

    geo_mech = load_mechanics_mesh(datadir)
    geo_ep = refine_geometry(geo_mech, 1)
    geo_ep.save(ep_outdir)
    return geo_ep


def main(outdir: str = "data"):

    outdir = Path(outdir)
    geo_ep = load_ep_mesh(outdir)

    return geo_ep


if __name__ == "__main__":
    main()
