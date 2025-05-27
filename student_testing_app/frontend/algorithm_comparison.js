async function loadAlgorithmStats() {
  const res = await fetch(`/admin/algorithms/stats?token=${token}`);
  const data = await res.json();

  const summary = data.summary;
  const algorithms = Object.keys(summary);
  const overallAccuracies = algorithms.map(a => summary[a].avg_overall_accuracy);
  const weakAccuracies = algorithms.map(a => summary[a].avg_weak_accuracy);
  const counts = algorithms.map(a => summary[a].count);

  // 🔀 Výpočet kombinovaného skóre
  const combinedScores = {};
  algorithms.forEach(algo => {
    const overall = summary[algo].avg_overall_accuracy || 0;
    const weak = summary[algo].avg_weak_accuracy || 0;
    combinedScores[algo] = 0.6 * overall + 0.4 * weak;
  });

  const bestOverall = algorithms.reduce((a, b) =>
    summary[a].avg_overall_accuracy > summary[b].avg_overall_accuracy ? a : b
  );
  const bestWeak = algorithms.reduce((a, b) =>
    summary[a].avg_weak_accuracy > summary[b].avg_weak_accuracy ? a : b
  );
  const bestCombined = algorithms.reduce((a, b) =>
    combinedScores[a] > combinedScores[b] ? a : b
  );

  new Chart(document.getElementById("overallAccuracyChart").getContext("2d"), {
    type: "bar",
    data: {
      labels: algorithms,
      datasets: [{
        label: "Priemerná celková presnosť správnej odpovede (%)",
        data: overallAccuracies,
        backgroundColor: "#2196f3"
      }]
    },
    options: {
      plugins: {
        title: { display: true, text: "Celková úspešnosť algoritmov" }
      },
      scales: {
        y: { beginAtZero: true, max: 100 }
      }
    }
  });

  new Chart(document.getElementById("weakAccuracyChart").getContext("2d"), {
    type: "bar",
    data: {
      labels: algorithms,
      datasets: [{
        label: "Presnosť správnej odpovede v slabých kategóriách (%)",
        data: weakAccuracies,
        backgroundColor: "#4caf50"
      }]
    },
    options: {
      plugins: {
        title: { display: true, text: "Úspešnosť v slabých kategóriách" }
      },
      scales: {
        y: { beginAtZero: true, max: 100 }
      }
    }
  });

  const summaryBox = document.getElementById("algo-stats-summary");
  summaryBox.innerHTML = `
    <p><strong>Počet študentov testovaných na algoritmoch:</strong></p>
    <ul>
      ${algorithms.map(algo => `<li>${algo}: ${summary[algo].count} študentov</li>`).join("")}
    </ul>
    <p><strong>Najlepší algoritmus podľa priemernej celkovej presnosti:</strong> ${bestOverall}</p>
    <p><strong>Najlepší algoritmus v zameraní na slabé kategórie:</strong> ${bestWeak}</p>
    <p><strong>Celkovo najefektívnejší algoritmus (aj v priemernej presnosti aj v zamerani na slabé kategórie):</strong> ${bestCombined}</p>
  `;

  const categoryDetailsBox = document.getElementById("algo-category-details");
  categoryDetailsBox.innerHTML = "<h3>🧩 Detail kategórií pre každý algoritmus:</h3>";

  algorithms.forEach(algo => {
    const stats = summary[algo];
    const categories = stats.category_stats || {};
    const labels = Object.keys(categories);
    const totalData = labels.map(cat => categories[cat].total || 0);
    const weakData = labels.map(cat => categories[cat].weak_count || 0);

    // ✅ Obal pre graf + tabuľku
    const wrapper = document.createElement("div");
    wrapper.style.marginBottom = "40px";

    // ✅ Plátno pre graf
    const canvas = document.createElement("canvas");
    canvas.id = `chart-${algo}`;
    canvas.style.marginTop = "15px";
    canvas.style.maxHeight = "250px";
    canvas.style.width = "100%";
    wrapper.appendChild(canvas);

    // ✅ Vytvorenie tabuľky
    const table = document.createElement("div");
    table.innerHTML = `
      <table border="1" style="border-collapse: collapse; margin-top: 10px;">
        <thead>
          <tr style="background:#f0f0f0;">
            <th colspan="3">${algo}</th>
          </tr>
          <tr>
            <th>Kategória</th>
            <th>Počet otázok</th>
            <th>Počet krát ako slabá</th>
          </tr>
        </thead>
        <tbody>
          ${Object.entries(categories).map(([cat, val]) => `
            <tr>
              <td>${cat}</td>
              <td>${val.total}</td>
              <td>${val.weak_count}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
    wrapper.appendChild(table);
    categoryDetailsBox.appendChild(wrapper);

    // ✅ Vykreslenie grafu
    const ctx = canvas.getContext("2d");
    new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Celkový počet otázok",
            data: totalData,
            backgroundColor: "rgba(33, 150, 243, 0.7)"
          },
        ]
      },
      options: {
        responsive: true,
        plugins: {
          title: {
            display: true,
            text: `Rozbor kategórií - ${algo}`
          }
        },
        scales: {
          y: { beginAtZero: true }
        }
      }
    });
  });
}

document.addEventListener("DOMContentLoaded", loadAlgorithmStats);