## Projekt to implementacja neuroewolucji 

Algorytm genetyczny służy jako optymalizator wag prostej sieci neuronowej typu feed-forward (MLP).

---

Sercem układu jest sieć o architekturze 8-16-4. Wejściem jest wektor stanu
ze środowiska Gymnasium (pozycja, prędkość liniowa i kątowa, orientacja oraz sensory dotyku nóg). Te dane przechodzą przez warstwę ukrytą z aktywacją ReLU, która wprowadza nieliniowość, i trafiają na warstwę wyjściową decydującą o akcji. Cała „inteligencja” tej sieci jest zawarta w wagach
i biasach, które tutaj nie są trenowane gradientowo
(jak w typowym backpropagation), ale są traktowane jako zestaw genów
(w algorytmie ewolucyjnym.

Algorytm PyGAD generuje populację 60 osobników, gdzie każdy jest reprezentowany przez długi wektor zmiennoprzecinkowy – to spłaszczone parametry sieci neuronowej. Ewaluacja odbywa się przez funkcję fitness,
która jest tutaj niestandardowa. Zamiast brać czysty wynik z gry, mamy logikę nagradzającą konkretne zachowania fizyczne: bonus za pierwszy kontakt
z podłożem oraz premię za utrzymanie pułapu po odbiciu. To wymusza na modelu specyficzną strategię lotu, a nie tylko najszybsze osadzenie na ziemi.

Używamy selekcji turniejowej, która wybiera najlepsze zestawy wag,
a następnie poddaje je krzyżowaniu rozproszonemu (scattered crossover).
Do tego dochodzi mutacja adaptacyjna – jej prawdopodobieństwo zmienia się w zależności od tego, czy populacja wpada w stagnację, co pozwala uniknąć lokalnych optimów. Dzięki parallel_processing na 16 wątkach, symulacja fizyczna dla dziesiątek osobników dzieje się równolegle, co pozwala na szybkie przejście przez 150 generacji. W efekcie, po zakończeniu cyklu, najlepszy wektor genów po "rozpakowaniu" z powrotem do macierzy wag sieci neuronowej daje gotowy kontroler, który w czasie rzeczywistym mapuje stany środowiska na optymalne wektory ciągu silników.
