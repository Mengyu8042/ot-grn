def test_double_ot_pipeline_runs():
    from ot_grn import generate_simulated_data, double_ot, evaluate_grn_accuracy

    # Step (i): Generate synthetic data
    exp1, exp2, true_plan = generate_simulated_data(
        p=500, n=100, diffgene=0.2, indegree=5
    )

    # Step (ii): Run Double OT
    ot_plan = double_ot(exp1, exp2, paired=False)

    # Step (iii): Evaluate the result
    auroc, aupr, ep = evaluate_grn_accuracy(true_plan, ot_plan)

    # Basic sanity checks
    assert ot_plan.shape[0] == ot_plan.shape[1] == 500
    assert auroc >= 0 and aupr >= 0 and ep >= 0

