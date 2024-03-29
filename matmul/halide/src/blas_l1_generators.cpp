/*
Copyright (c) 2012-2020 MIT CSAIL, Google, Facebook, Adobe, NVIDIA CORPORATION, and other contributors.

Developed by:

  The Halide team
  http://halide-lang.org

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "Halide.h"
#include <vector>

using namespace Halide;

namespace {

// Generator class for BLAS axpy operations.
template<class T>
class AXPYGenerator : public Generator<AXPYGenerator<T>> {
public:
    typedef Generator<AXPYGenerator<T>> Base;
    using Base::get_target;
    using Base::natural_vector_size;
    using Base::target;
    template<typename T2>
    using Input = typename Base::template Input<T2>;
    template<typename T2>
    using Output = typename Base::template Output<T2>;

    GeneratorParam<bool> vectorize_ = {"vectorize", true};
    GeneratorParam<int> block_size_ = {"block_size", 1024};
    GeneratorParam<bool> scale_x_ = {"scale_x", true};
    GeneratorParam<bool> add_to_y_ = {"add_to_y", true};

    // Standard ordering of parameters in AXPY functions.
    Input<T> a_ = Input<T>{"a", 1};
    Input<Buffer<T>> x_ = Input<Buffer<T>>{"x", 1};
    Input<Buffer<T>> y_ = Input<Buffer<T>>{"y", 1};

    Output<Buffer<T>> result_ = Output<Buffer<T>>{"result", 1};

    template<class Arg>
    Expr calc(Arg i) {
        if (static_cast<bool>(scale_x_) && static_cast<bool>(add_to_y_)) {
            return a_ * x_(i) + y_(i);
        } else if (static_cast<bool>(scale_x_)) {
            return a_ * x_(i);
        } else {
            return x_(i);
        }
    }

    void generate() {
        assert(get_target().has_feature(Target::NoBoundsQuery));

        const int vec_size = vectorize_ ? natural_vector_size(type_of<T>()) : 1;
        Expr size = x_.width();
        Expr size_vecs = (size / vec_size) * vec_size;
        Expr size_tail = size - size_vecs;

        Var i("i");
        RDom vecs(0, size_vecs, "vec");
        RDom tail(size_vecs, size_tail, "tail");
        result_(i) = undef(type_of<T>());
        result_(vecs) = calc(vecs);
        result_(tail) = calc(tail);

        if (vectorize_) {
            Var ii("ii");
            result_.update().vectorize(vecs, vec_size);
        }

        result_.bound(i, 0, x_.width());
        result_.dim(0).set_bounds(0, x_.width());

        x_.dim(0).set_min(0);
        y_.dim(0).set_bounds(0, x_.width());
    }
};

// Generator class for BLAS dot operations.
template<class T>
class DotGenerator : public Generator<DotGenerator<T>> {
public:
    typedef Generator<DotGenerator<T>> Base;
    using Base::get_target;
    using Base::natural_vector_size;
    using Base::target;
    template<typename T2>
    using Input = typename Base::template Input<T2>;
    template<typename T2>
    using Output = typename Base::template Output<T2>;

    GeneratorParam<bool> vectorize_ = {"vectorize", true};
    GeneratorParam<bool> parallel_ = {"parallel", true};
    GeneratorParam<int> block_size_ = {"block_size", 1024};

    Input<Buffer<T>> x_ = Input<Buffer<T>>{"x", 1};
    Input<Buffer<T>> y_ = Input<Buffer<T>>{"y", 1};

    Output<Buffer<T>> result_ = Output<Buffer<T>>{"result", 0};

    void generate() {
        assert(get_target().has_feature(Target::NoBoundsQuery));

        const int vec_size = vectorize_ ? natural_vector_size(type_of<T>()) : 1;
        Expr size = x_.width();
        Expr size_vecs = size / vec_size;
        Expr size_tail = size - size_vecs * vec_size;

        Var i("i");
        if (vectorize_) {
            Func dot;

            RDom k(0, size_vecs);
            dot(i) += x_(k * vec_size + i) * y_(k * vec_size + i);

            RDom lanes(0, vec_size);
            RDom tail(size_vecs * vec_size, size_tail);
            result_() = sum(dot(lanes));
            result_() += sum(x_(tail) * y_(tail));

            dot.compute_root().vectorize(i);
            dot.update(0).vectorize(i);
        } else {
            RDom k(0, size);
            result_() = sum(x_(k) * y_(k));
        }

        x_.dim(0).set_bounds(0, size);
        y_.dim(0).set_bounds(0, size);
    }
};

// Generator class for BLAS dot operations.
template<class T>
class AbsSumGenerator : public Generator<AbsSumGenerator<T>> {
public:
    typedef Generator<AbsSumGenerator<T>> Base;
    using Base::get_target;
    using Base::natural_vector_size;
    using Base::target;
    template<typename T2>
    using Input = typename Base::template Input<T2>;
    template<typename T2>
    using Output = typename Base::template Output<T2>;

    GeneratorParam<bool> vectorize_ = {"vectorize", true};
    GeneratorParam<bool> parallel_ = {"parallel", true};
    GeneratorParam<int> block_size_ = {"block_size", 1024};

    Input<Buffer<T>> x_ = Input<Buffer<T>>{"x", 1};

    Output<Buffer<T>> result_ = Output<Buffer<T>>{"result", 0};

    void generate() {
        assert(get_target().has_feature(Target::NoBoundsQuery));

        const int vec_size = vectorize_ ? natural_vector_size(type_of<T>()) : 1;
        Expr size = x_.width();
        Expr size_vecs = size / vec_size;
        Expr size_tail = size - size_vecs * vec_size;

        Var i("i");
        if (vectorize_) {
            Func norm;

            RDom k(0, size_vecs);
            norm(i) += abs(x_(k * vec_size + i));

            RDom lanes(0, vec_size);
            RDom tail(size_vecs * vec_size, size_tail);
            result_() = sum(norm(lanes));
            result_() += sum(abs(x_(tail)));

            norm.compute_root().vectorize(i);
            norm.update(0).vectorize(i);
        } else {
            RDom k(0, x_.width());
            result_() = sum(abs(x_(k)));
        }

        x_.dim(0).set_min(0);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(AXPYGenerator<float>, saxpy)
HALIDE_REGISTER_GENERATOR(AXPYGenerator<double>, daxpy)
HALIDE_REGISTER_GENERATOR(DotGenerator<float>, sdot)
HALIDE_REGISTER_GENERATOR(DotGenerator<double>, ddot)
HALIDE_REGISTER_GENERATOR(AbsSumGenerator<float>, sasum)
HALIDE_REGISTER_GENERATOR(AbsSumGenerator<double>, dasum)
