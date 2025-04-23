def test_gastric_cancer_pipeline_runs():
    from ot_grn import load_gastric_cancer_data, double_ot, extract_top_edges

    # Step (i): Load paired normal and tumor samples
    exp1, exp2 = load_gastric_cancer_data(same_source=True, paired=True)

    # Step (ii): Run Double OT
    ot_plan = double_ot(exp1, exp2)

    # Step (iii): Extract top regulatory edges and key genes
    gene_names = exp1.index.tolist()
    top_edges, key_genes = extract_top_edges(ot_plan, gene_names, num=500)

    # Basic sanity checks
    assert ot_plan.shape[0] == ot_plan.shape[1] == len(gene_names)
    assert len(top_edges) <= 500
    assert len(key_genes) > 0
