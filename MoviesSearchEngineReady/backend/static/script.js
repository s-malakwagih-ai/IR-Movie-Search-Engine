let charts = {};

function value(id){return document.getElementById(id).value.trim();}

async function runSearch(){
    const q = value("query");
    const resultsBox = document.getElementById("results");
    const info = document.getElementById("queryInfo");
    if(!q){resultsBox.innerHTML = '<div class="empty">Write a search query first.</div>';return;}
    resultsBox.innerHTML = '<div class="empty">Searching...</div>';
    const params = new URLSearchParams({
        q:q,
        type:value("typeFilter"),
        year_from:value("yearFrom"),
        year_to:value("yearTo"),
        min_rating:value("minRating") || "0",
        expand:document.getElementById("expand").checked
    });
    const res = await fetch(`/api/search?${params.toString()}`);
    const data = await res.json();
    info.style.display = "block";
    info.innerHTML = `<b>Processed query:</b> ${data.processed_query || "-"}<br><b>Expanded terms:</b> ${data.expanded_terms.join(", ") || "-"}<br><b>Results:</b> ${data.count}`;
    if(data.results.length === 0){
        resultsBox.innerHTML = '<div class="empty">No matching results found. Try another query or remove filters.</div>';
        return;
    }
    resultsBox.innerHTML = data.results.map(item => `
        <div class="result-card">
            <div class="result-top">
                <div><span class="rank">#${item.rank}</span> <b>${item.title}</b></div>
                <div class="score">Score ${item.score}</div>
            </div>
            <div class="meta">${item.type} • ${item.year} • Rating ${item.rating}</div>
            <div class="abstract">${item.abstract || "No abstract available."}</div>
        </div>
    `).join("");
}

async function getRecommendations(){
    const title = value("recommendTitle");
    const box = document.getElementById("recommendations");
    if(!title){box.innerHTML = '<div class="empty">Enter a title first.</div>';return;}
    box.innerHTML = '<div class="empty">Finding similar movies...</div>';
    const res = await fetch(`/api/recommend?title=${encodeURIComponent(title)}`);
    const data = await res.json();
    if(data.recommendations.length === 0){box.innerHTML = '<div class="empty">No similar movies found. Check the title spelling.</div>';return;}
    box.innerHTML = data.recommendations.map(item => `
        <div class="mini-card"><b>${item.title}</b><div class="meta">${item.year} • Rating ${item.rating} • Similarity ${item.similarity}</div></div>
    `).join("");
}

function drawChart(id, type, labels, values, label){
    if(charts[id]) charts[id].destroy();
    charts[id] = new Chart(document.getElementById(id), {
        type:type,
        data:{labels:labels,datasets:[{label:label,data:values}]},
        options:{responsive:true,plugins:{legend:{display:type !== "bar"}},scales:type === "pie" ? {} : {y:{beginAtZero:true}}}
    });
}

async function loadAnalytics(){
    const res = await fetch('/api/analytics');
    const data = await res.json();
    drawChart("typeChart", "pie", Object.keys(data.type_counts), Object.values(data.type_counts), "Type Count");
    drawChart("ratingChart", "bar", Object.keys(data.rating_bins), Object.values(data.rating_bins), "Rating Distribution");
    drawChart("yearChart", "line", Object.keys(data.yearly_counts), Object.values(data.yearly_counts), "Yearly Documents");

    document.getElementById("topRated").innerHTML = data.top_rated.map(item => `
        <div class="data-row"><b>${item.title}</b><div class="meta">${item.type} • ${item.year} • ${Number(item.rating).toFixed(2)}</div></div>
    `).join("");

    document.getElementById("clusters").innerHTML = data.clusters.map(c => `
        <div class="data-row"><div class="cluster-title">Cluster ${c.cluster}</div><div>${c.count} documents • Avg rating ${c.avg_rating} • Avg year ${c.avg_year}</div><small>${c.top_titles.join(" | ")}</small></div>
    `).join("");

    const net = await fetch('/api/network');
    const network = await net.json();
    document.getElementById("network").innerHTML = `
        <div class="data-row"><b>Nodes:</b> ${network.nodes_count}</div>
        <div class="data-row"><b>Edges:</b> ${network.edges_count}</div>
        <div class="data-row"><b>Important terms:</b><br>${network.important_text_terms.slice(0,12).join(", ")}</div>
    `;
}

window.addEventListener("load", () => {
    document.getElementById("query").addEventListener("keydown", e => { if(e.key === "Enter") runSearch(); });
    loadAnalytics();
});
