const $ = id => document.getElementById(id);
const percent = value => (value * 100).toFixed(2) + '%';
const format = new Intl.NumberFormat();
const MODELS = ['LightGBM Tweedie', 'MLP Poisson', 'Piecewise monotone', 'Naive last week'];

function element(tag, value, className) {
  const item = document.createElement(tag);
  item.textContent = value;
  if (className) item.className = className;
  return item;
}
function metric(name, value) {
  const card = element('div', '', 'metric');
  card.append(element('strong', value), element('small', name));
  return card;
}
function bar(name, value, maximum) {
  const row = element('div', '', 'bar');
  const gauge = document.createElement('progress');
  gauge.max = maximum;
  gauge.value = value;
  gauge.setAttribute('aria-label', name);
  row.append(element('span', name), gauge, element('strong', percent(value)));
  return row;
}
function draw(scenarios, index) {
  const canvas = $('curve');
  const {width, height} = canvas.getBoundingClientRect();
  if (!width || !height) return;
  const ratio = devicePixelRatio || 1;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const ctx = canvas.getContext('2d');
  ctx.scale(ratio, ratio);
  const left = 31, right = width - 8, top = 9, bottom = height - 19;
  const max = Math.ceil(Math.max(...scenarios.flatMap(p => [p.tweedie, p.monotone])));
  const x = i => left + (right - left) * i / 60;
  const y = v => bottom - (bottom - top) * v / (max || 1);
  ctx.font = '10px system-ui';
  ctx.fillStyle = '#536477';
  for (const value of [0, max / 2, max]) {
    ctx.fillText(String(value), 0, y(value) + 3);
    ctx.strokeStyle = '#e3e9f0';
    ctx.beginPath();
    ctx.moveTo(left, y(value));
    ctx.lineTo(right, y(value));
    ctx.stroke();
  }
  ctx.fillText('0%', left, height - 4);
  ctx.fillText('60%', right - 21, height - 4);
  ctx.strokeStyle = '#9baaba';
  ctx.beginPath();
  ctx.moveTo(x(index), top);
  ctx.lineTo(x(index), bottom);
  ctx.stroke();
  for (const [key, color] of [['tweedie', '#702c46'], ['monotone', '#305e56']]) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    scenarios.forEach((point, i) => i ? ctx.lineTo(x(i), y(point[key])) : ctx.moveTo(x(i), y(point[key])));
    ctx.stroke();
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.arc(x(index), y(scenarios[index][key]), 3, 0, 2 * Math.PI);
    ctx.fill();
  }
}
function initialise(report) {
  const selected = MODELS.map(name => report.summary.find(row => row.model === name));
  $('metrics').hidden = false;
  $('mode').textContent = 'Historical evaluation';
  $('metrics').replaceChildren(metric('Historical test rows', format.format(report.rows)),
    metric('Tweedie mean WAPE · 3 runs', percent(selected[0].wape)),
    metric('Last-week baseline WAPE', percent(selected[3].wape)));
  $('models').replaceChildren(...selected.map(row => bar(row.model, row.wape, Math.max(1, ...selected.map(model => model.wape)))));
  $('support-bars').replaceChildren(...report.support.map(row => bar(row.discount + '%', row.outside, 1)));
  const selector = $('support-discount');
  if (report.support.length) {
    selector.replaceChildren(...report.support.map(row => new Option(row.discount + '% off', String(row.discount))));
    const updateSupport = () => {
      const row = report.support.find(item => String(item.discount) === selector.value);
      $('support-detail').textContent = row.discount + '% off · ' + percent(row.outside) + ' of test contexts outside the threshold';
    };
    selector.addEventListener('change', updateSupport);
    $('support-controls').hidden = false;
    $('support-detail').hidden = false;
    updateSupport();
  }
  if (!report.scenarios) return;
  $('mode').textContent = 'Local checkpoint replay';
  $('unavailable').hidden = true;
  $('replay').hidden = false;
  const data = report.scenarios;
  $('context').textContent = 'Forecast week beginning ' + data.forecast_date + ' · observed ' +
    format.format(data.observed_units) + (data.observed_units === 1 ? ' unit at ' : ' units at ') +
    percent(data.observed_discount) + ' discount';
  const slider = $('discount');
  const render = () => {
    const index = Number(slider.value);
    const scenario = data.forecast[index];
    $('level').textContent = scenario.discount + '%';
    $('predictions').replaceChildren(metric('Tweedie · units · burgundy', scenario.tweedie.toFixed(2)),
      metric('Monotone network · units · green', scenario.monotone.toFixed(2)));
    draw(data.forecast, index);
  };
  slider.addEventListener('input', render);
  new ResizeObserver(render).observe($('curve'));
  render();
}

fetch('/api/study').then(response => {
  if (!response.ok) throw new Error('Unable to load study data.');
  return response.json();
}).then(initialise).catch(error => {
  $('mode').textContent = 'Study unavailable';
  $('error').textContent = error.message;
  $('error').hidden = false;
});
