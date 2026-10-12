// Turns the full question list into one question per screen.
// Without JS the page still works: every question shows and the form posts as normal.
(function () {
  const form = document.querySelector("[data-screen]");
  if (!form) return;

  const questions = Array.from(form.querySelectorAll("[data-question]"));
  const progress = form.querySelector("[data-progress]");
  const current = form.querySelector("[data-current]");
  const cells = Array.from(form.querySelectorAll(".progress-track li"));
  const back = form.querySelector("[data-back]");
  const next = form.querySelector("[data-next]");
  const submit = form.querySelector("[data-submit]");

  // If the server sent us back with an error, jump to the first unanswered question.
  let index = questions.findIndex((q) => !q.querySelector("input:checked"));
  if (index === -1) index = questions.length - 1;
  if (!form.querySelector(".error")) index = 0;

  form.classList.add("is-stepped");
  progress.hidden = false;
  back.hidden = false;

  const answered = (q) => Boolean(q.querySelector("input:checked"));

  function render(moveFocus) {
    questions.forEach((q, i) => q.classList.toggle("is-current", i === index));
    cells.forEach((c, i) => {
      c.classList.toggle("here", i === index);
      c.classList.toggle("done", i !== index && answered(questions[i]));
    });
    current.textContent = index + 1;

    const last = index === questions.length - 1;
    back.disabled = index === 0;
    back.hidden = index === 0;
    next.hidden = last;
    submit.hidden = !last;
    next.disabled = !answered(questions[index]);
    submit.disabled = !questions.every(answered);

    if (moveFocus) {
      const target = questions[index].querySelector("input:checked") || questions[index].querySelector("input");
      target.focus({ preventScroll: true });
      window.scrollTo({ top: 0 });
    }
  }

  form.addEventListener("change", () => render(false));
  next.addEventListener("click", () => {
    if (index < questions.length - 1) { index += 1; render(true); }
  });
  back.addEventListener("click", () => {
    if (index > 0) { index -= 1; render(true); }
  });

  // Enter on a chosen answer moves forward, like pressing Next.
  form.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" || e.target.type !== "radio") return;
    e.preventDefault();
    if (index < questions.length - 1 && answered(questions[index])) { index += 1; render(true); }
    else if (questions.every(answered)) form.requestSubmit();
  });

  render(false);
})();
