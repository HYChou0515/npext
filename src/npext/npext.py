import numpy as np


class ExtendedNDArray(np.ndarray):
    def __new__(cls, input_array, *args, **kwargs):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        print(f"{method=}")
        cls = self.__class__
        args = []
        in_no = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, cls):
                in_no.append(i)
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = out
        out_no = []
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, cls):
                    out_no.append(j)
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            return

        if method == "__call__":
            def cast_back_to_class(result):
                # do not cast
                return np.asarray(result).view(cls)
        else:
            def cast_back_to_class(result):
                # do not cast
                return result

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((cast_back_to_class(result)
                         if output is None else output)
                        for result, output in zip(results, outputs))

        return results[0] if len(results) == 1 else results


class Image(ExtendedNDArray):
    def __new__(cls, *args, **kwargs):
        print("Image new")
        return super().__new__(cls, *args, **kwargs)

    def __array_finalize__(self, obj):
        print(f"finalize {obj}")
        if obj is None: return
        print(f"finalize1 {obj}")

        if len(self.shape) == 2:
            print(f"finalize2 {obj}")
            return obj
        else:
            print(f"finalize3 {obj}")
            return 3

    @property
    def width(self):
        return self.shape[1]

    @property
    def height(self):
        return self.shape[0]

image = Image(np.arange(20).reshape((4, 5)))
arr = np.arange(20).reshape((4, 5))
