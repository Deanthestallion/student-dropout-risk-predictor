import nbformat
nb = nbformat.read("notebook.ipynb", as_version=4)
errs = [(i, o.ename, o.evalue) for i, c in enumerate(nb.cells) if c.cell_type == "code"
        for o in c.get("outputs", []) if o.output_type == "error"]
print("ERROR CELLS:", errs if errs else "none")
fails = [l for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
         for l in o.get("text", "").splitlines() if "[FAIL]" in l]
print("FAILED CHECKS:", fails if fails else "none")
npass = sum(1 for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
            for l in o.get("text", "").splitlines() if "[PASS]" in l)
print("PASSED CHECKS:", npass)
print("executed code cells:", sum(1 for c in nb.cells if c.cell_type == "code" and c.get("execution_count")))
print("images embedded:", sum(1 for c in nb.cells for o in c.get("outputs", [])
                              if "image/png" in o.get("data", {})))
