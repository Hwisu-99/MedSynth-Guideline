import time
import tracemalloc


class MeasureResource:
    """합성 데이터 생성 시간 및 메모리 사용량 측정 컨텍스트 매니저.

    사용 예시:
        with MeasureResource("SDV 학습"):
            synthesizer.fit(df)

        with MeasureResource("SDV 생성", n_rows=10000) as m:
            df_syn = synthesizer.sample(10000)
        print(m.elapsed, m.peak_mb)
    """

    def __init__(self, label: str = "", n_rows: int = None):
        self.label   = label
        self.n_rows  = n_rows
        self.elapsed = None
        self.peak_mb = None

    def __enter__(self):
        tracemalloc.start()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_mb = peak_bytes / 1024 / 1024
        self._print()

    def _print(self):
        mins, secs = divmod(self.elapsed, 60)
        header = f"[{self.label}]" if self.label else "[측정 결과]"
        print(f"\n{header}")
        print(f"  소요 시간  : {self.elapsed:.1f}초  ({int(mins)}분 {secs:.1f}초)")
        print(f"  피크 메모리: {self.peak_mb:.1f} MB")
        if self.n_rows:
            print(f"  행당 시간  : {self.elapsed / self.n_rows * 1000:.3f} ms/행  "
                  f"({self.n_rows:,}행 기준)")