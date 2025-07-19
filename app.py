from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from scipy import stats

app = Flask(__name__)
CORS(app)  # ✅ This enables CORS for all routes


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()

    param_names = data["paramNames"]
    lsl_values = data["lslValues"]
    usl_values = data["uslValues"]
    batches = data["batches"]
    lock_point = data["lockPoint"]

    batch_array = np.array(data["batches"], dtype=float)

    if batch_array.ndim == 0:
        return jsonify(error="No batch data received or incorrect structure."), 400

    elif batch_array.ndim == 1:
        batch_array = batch_array.reshape(1, -1)

    elif batch_array.ndim > 2:
        return jsonify(error="Batch data is too deeply nested. Please check your Excel or input format."), 400

    try:
        num_batches, num_params = batch_array.shape
    except ValueError as ve:
        return jsonify(error=f"Unable to extract batch dimensions: {ve}"), 400

    num_params = len(param_names)
    num_batches = len(batch_array)

    # Step 1: Build global in-spec mask (a batch is valid only if all param values are in-spec)
    in_spec_batch_mask = np.ones(num_batches, dtype=bool)
    for i in range(num_params):
        values = batch_array[:, i]
        in_spec_batch_mask &= (values >= lsl_values[i]) & (values <= usl_values[i])

    in_spec_indexes = np.where(in_spec_batch_mask)[0]

    # Step 2: Check global locking point condition
    if len(in_spec_indexes) < lock_point:
        return jsonify({
            "error": f"Only {len(in_spec_indexes)} valid (in-specification) batches found across all parameters. Lock point requires {lock_point}."
        }), 400

    # Prepare HTML tables
    metrics_html_baseline = ['<h3>SQC Metrics Summary – Baseline Data Control Limits</h3><table><tr><th>Parameter</th><th>Mean</th><th>SD</th><th>%RSD</th><th>LCL</th><th>UCL</th><th>CpK</th></tr>']
    metrics_html_rolling = [
  '<h3>SQC Metrics Summary – Rolling Data</h3>',
  '<table><tr><th>Parameter</th><th>Mean</th><th>SD</th><th>%RSD</th><th>CpK</th></tr>'
]
   
    charts = []

    for i in range(num_params):
        values = batch_array[:, i]
        lsl = lsl_values[i]
        usl = usl_values[i]

        baseline_indexes = in_spec_indexes[:lock_point]
        baseline_values = values[baseline_indexes]

        # ✅ Global in-spec batches only
        rolling_values = values[in_spec_batch_mask & ~np.isnan(values)]
        
        def mean_sd(a):
            m = np.mean(a)
            s = np.std(a, ddof=1)
            return m, s

        if len(baseline_values) >= 2:
            m, s = mean_sd(baseline_values)
            lcl, ucl = m - 3 * s, m + 3 * s
            cpk = min((usl - m) / (3 * s), (m - lsl) / (3 * s)) if s else 0
            metrics_html_baseline.append(f"<tr><td>{param_names[i]}</td><td>{m:.2f}</td><td>{s:.2f}</td><td>{s/m*100:.2f}</td><td>{lcl:.2f}</td><td>{ucl:.2f}</td><td>{cpk:.2f}</td></tr>")
        else:
            m, s, lcl, ucl, cpk = 0, 0, 0, 0, 0
            metrics_html_baseline.append(f"<tr><td>{param_names[i]}</td><td colspan='6'>Insufficient baseline data</td></tr>")

        if len(rolling_values) >= 2:
            rm, rs = mean_sd(rolling_values)
            rlcl, rucl = rm - 3 * rs, rm + 3 * rs
            rcpk = min((usl - rm) / (3 * rs), (rm - lsl) / (3 * rs)) if rs else 0
            metrics_html_rolling.append(f"<tr><td>{param_names[i]}</td><td>{rm:.2f}</td><td>{rs:.2f}</td><td>{rs/rm*100:.2f}</td><td>{rcpk:.2f}</td></tr>")
        else:
            metrics_html_rolling.append(f"<tr><td>{param_names[i]}</td><td colspan='6'>Insufficient rolling data</td></tr>")

        # Shift/Drift detection
        shift_flags = [False] * num_batches
        drift_flags = [False] * num_batches
        for j in range(num_batches - 8):
            seg = values[j:j + 9]
            if np.all(seg > m) or np.all(seg < m):
                for k in range(j, j + 9):
                    shift_flags[k] = True
        for j in range(num_batches - 5):
            seg = values[j:j + 6]
            if np.all(np.diff(seg) > 0) or np.all(np.diff(seg) < 0):
                for k in range(j, j + 6):
                    drift_flags[k] = True

        charts.append({
            "parameter": param_names[i],
            "values": values.tolist(),
            "mean": m,
            "lcl": lcl,
            "ucl": ucl,
            "lsl": lsl,
            "usl": usl,
            "shiftFlags": shift_flags,
            "driftFlags": drift_flags
        })

    metrics_html_baseline.append('</table>')
    metrics_html_rolling.append('</table>')

    return jsonify({
        "metricsHtml": ''.join(metrics_html_baseline + metrics_html_rolling),
        "charts": charts
    })



if __name__ == "__main__":
    app.run(debug=True)
