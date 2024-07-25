

#undef NDEBUG

#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/statistics.h>

#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <mutex>
#include <optional>
#include <thread>
#include <type_traits>
#include <omp.h>


MTS_NAMESPACE_BEGIN

double statsSdtreeBuild = 0.0;
double statsSdtreeReset = 0.0;
double statsPhaseTimeRendering = 0.0;
double statsPhaseTimeRenderPass = 0.0;
double statsPhaseTimeTotal = 0.0;
double statsPhaseTimeSampleMat = 0.0;
double statsPhaseTimeCommit = 0.0;
double statsPhaseTimeRenderBlockSum = 0.0;
double statsPhaseTimeRenderPostproc = 0.0;
int64_t statsSuperfuseDFSCall = 0;
int64_t statsSuperfusePushdownCall = 0;
int64_t statsResetBFSCall = 0;
int64_t statsCommitCall = 0;
int64_t statsCommitRequestTotal = 0;

void printMystats() {
}


class HDTimer {
public:
    using Unit = std::chrono::nanoseconds;

    HDTimer() {
        start = std::chrono::system_clock::now();
    }

    double value() const {
        auto now = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<Unit>(now - start);
        return (double)duration.count() * 1e-9;
    }

    double reset() {
        auto now = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<Unit>(now - start);
        start = now;
        return (double)duration.count() * 1e-9;
    }

private:
    std::chrono::system_clock::time_point start;
};


Float computeElapsedSeconds(std::chrono::steady_clock::time_point start) {
    auto current = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
    return (Float)ms.count() / 1000;
}

float global_temp_param = 0;
float global_guiding_mis_weight_ad = 0;
float global_decay_rate = 1.0;
int g_skip_dtree_build = 0;
int g_stepper = 0;

struct StepperRawSample {
    Point3f pos;
    Vector3f dir;
    Float val;
};

class BlobWriter {
public:
    BlobWriter(const std::string& filename)
        : f(filename, std::ios::out | std::ios::binary) {
    }

    template <typename Type>
    typename std::enable_if<std::is_standard_layout<Type>::value, BlobWriter&>::type
        operator << (Type Element) {
        Write(&Element, 1);
        return *this;
    }

    // CAUTION: This function may break down on big-endian architectures.
    //          The ordering of bytes has to be reverted then.
    template <typename T>
    void Write(T* Src, size_t Size) {
        f.write(reinterpret_cast<const char*>(Src), Size * sizeof(T));
    }

private:
    std::ofstream f;
};


class Pfstream {
public:
    explicit Pfstream(char* filename) :
        valid{ false }, printfFile{ nullptr }, buf{ nullptr }, last(nullptr), bufsize(1024){//attention bufsize < 8 is dangerous
        printfFile = std::fopen(filename, "w");
        valid = true;
        if (printfFile == NULL) {
            SLog(ELogLevel::EError, "cannot open file: %s", filename);
            SAssert(false);
            valid = false;
        }
        buf = new char[bufsize];
        last = buf + bufsize;
        out = buf;
    }
    ~Pfstream() {
        consume();
        if (valid) {
            std::fclose(printfFile);
        }
        delete[] buf;
        buf = nullptr;
        last = nullptr;
        out = nullptr;
    }
    void consume() {
        if (out >= buf) {
            *out = '\0';
            fprintf(printfFile, "%s", buf);
            out = buf;
        }
    }

    void close() {
        consume();
        if (valid) {
            std::fclose(printfFile);
            valid = false;
        }
    }

    template<class T>
    void produce(const char* format, T& t) {
        //1. write anyway regardless of the buf size
        int nw = snprintf(out, last - out, format, t);//nw: strlen
        long long  res = last - out;
        if (res <= nw) {
            //2. if not enough abandon, and do consume first
            consume();
            //then write again
            nw = snprintf(out, last - out, format, t);
        }
        //3. update out pointer
        out += nw;
    }
    void resize(int newsize) {
        char* new_buf = new char[newsize];
        char* new_last = new_buf + newsize;

        char* tmp = buf;

        buf = new_buf;
        last = new_last;
        out = buf;
        bufsize = newsize;

        delete[] tmp;
    }
    void produce(const char* s) {
        //1. write anyway regardless of the buf size
        int nw = snprintf(out, last - out, s);//nw: strlen
        //1. test the empty buffer size vs data size need to write
        //if nw is bigger than buf size, do enlarge.
        if (nw >= bufsize) {
            consume();
            resize(2 * nw);
            //must recurse
            produce(s);
            return;
        }
        long long res = last - out;
        if (res <= nw) {
            //2. if not enough ,abandon, and do consume first
            consume();
            //then write again
            nw = snprintf(out, last - out, s);
        }
        //3. update out pointer
        out += nw;
    }

    inline Pfstream& operator<<(const float f) {
        produce("%g", f);
        return *this;
    }

    inline Pfstream& operator<<(const int i) {
        produce("%d", i);
        return *this;
    }
    inline Pfstream& operator<<(const size_t i) {
        produce("%u", i);
        return *this;
    }

    inline Pfstream& operator<<(const char c) {
        produce("%c", c);
        return *this;
    }

    inline Pfstream& operator<<(std::basic_ostream<char>& (*pf)(std::basic_ostream<char>&)) {
        //Here we assume pf is 'endl'
        produce("\n");
        return *this;
    }

    inline Pfstream& operator<<(const char* s) {
        produce(s);
        return *this;
    }

    inline Pfstream& operator<<(const double d) {
        produce("%g", d);
        return *this;
    }

private:
    FILE* printfFile;
    bool valid;
    char* buf, * last, * out;
    int bufsize;
};
class Pstringstream {
public:
    Pstringstream() :
        buf{ nullptr }, last(nullptr), bufsize(64){//attention bufsize < 8 is dangerous
        buf = new char[bufsize];
        last = buf + bufsize;
        out = buf;
    }
    ~Pstringstream() {
        delete[] buf;
        buf = nullptr;
        last = nullptr;
        out = nullptr;
    }
    void resize(int newsize) {
        //std::cout << "resize to :" << newsize << std::endl;
        char* new_buf = new char[newsize];
        char* new_last = new_buf + newsize;

        char* tmp = buf;

        memcpy(new_buf, buf, bufsize);

        long long offset = out - buf;
        buf = new_buf;
        last = new_last;
        out = new_buf + offset;
        bufsize = newsize;

        delete[] tmp;
    }

    template<class T>
    void produce(const char* format, T& t) {
        //1. write anyway regardless of the buf size
        int nw = snprintf(out, last - out, format, t);//nw: strlen
        long long  res = last - out;
        if (res <= nw) {
            while (res <= nw) {
                //2. if not enough abandon, resize,and recurse
                resize(bufsize * 4);
                res = last - out;
            }
            nw = snprintf(out, last - out, format, t);
        }
        //3. update out pointer
        out += nw;
    }
   
    void produce(const char* s) {
        //1. write anyway regardless of the buf size
        int nw = snprintf(out, last - out, s);//nw: strlen
        long long  res = last - out;
        if (res <= nw) {
            while (res <= nw) {
                //2. if not enough abandon, resize,and recurse
                resize(bufsize * 2);
                res = last - out;
            }
            nw = snprintf(out, last - out, s);
        }
        //3. update out pointer
        out += nw;
    }

    inline Pstringstream& operator<<(const float f) {
        produce("%g", f);
        return *this;
    }

    inline Pstringstream& operator<<(const int i) {
        produce("%d", i);
        return *this;
    }
    inline Pstringstream& operator<<(const size_t i) {
        produce("%u", i);
        return *this;
    }

    inline Pstringstream& operator<<(const char c) {
        produce("%c", c);
        return *this;
    }

    inline Pstringstream& operator<<(std::basic_ostream<char>& (*pf)(std::basic_ostream<char>&)) {
        //Here we assume pf is 'endl'
        produce("\n");
        return *this;
    }

    inline Pstringstream& operator<<(const char* s) {
        produce(s);
        return *this;
    }

    inline Pstringstream& operator<<(const double d) {
        produce("%g", d);
        return *this;
    }
    inline Pstringstream& operator<<(const std::string& s) {
        produce(s.c_str());
        return *this;
    }

    std::string str() {
        std::string result(buf);
        return result;
    }

private:
    char* buf, * last, * out;
    int bufsize;
};

//use typedef OFSTREAM for flexibility
typedef Pfstream OFSTREAM;
//typedef std::ofstream OFSTREAM;
typedef Pstringstream SSTREAM;
//typedef std::stringstream SSTREAM;


static void addToAtomicFloat(std::atomic<Float>& var, Float val) {
    auto current = var.load();
    while (!var.compare_exchange_weak(current, current + val));
}

inline Float logistic(Float x) {
    return 1 / (1 + std::exp(-x));
}

// Implements the stochastic-gradient-based Adam optimizer [Kingma and Ba 2014]
class AdamOptimizer {
public:
    AdamOptimizer(Float learningRate, int batchSize = 1, Float epsilon = 1e-08f, Float beta1 = 0.9f, Float beta2 = 0.999f) {
        m_hparams = { learningRate, batchSize, epsilon, beta1, beta2 };
    }

    AdamOptimizer& operator=(const AdamOptimizer& arg) {
        m_state = arg.m_state;
        m_hparams = arg.m_hparams;
        return *this;
    }

    AdamOptimizer(const AdamOptimizer& arg) {
        *this = arg;
    }

    void append(Float gradient, Float statisticalWeight) {
        m_state.batchGradient += gradient * statisticalWeight;
        m_state.batchAccumulation += statisticalWeight;

        if (m_state.batchAccumulation > m_hparams.batchSize) {
            step(m_state.batchGradient / m_state.batchAccumulation);

            m_state.batchGradient = 0;
            m_state.batchAccumulation = 0;
        }
    }

    void step(Float gradient) {
        ++m_state.iter;

        Float actualLearningRate = m_hparams.learningRate * std::sqrt(1 - std::pow(m_hparams.beta2, m_state.iter)) / (1 - std::pow(m_hparams.beta1, m_state.iter));
        m_state.firstMoment = m_hparams.beta1 * m_state.firstMoment + (1 - m_hparams.beta1) * gradient;
        m_state.secondMoment = m_hparams.beta2 * m_state.secondMoment + (1 - m_hparams.beta2) * gradient * gradient;
        m_state.variable -= actualLearningRate * m_state.firstMoment / (std::sqrt(m_state.secondMoment) + m_hparams.epsilon);

        // Clamp the variable to the range [-20, 20] as a safeguard to avoid numerical instability:
        // since the sigmoid involves the exponential of the variable, value of -20 or 20 already yield
        // in *extremely* small and large results that are pretty much never necessary in practice.
        m_state.variable = std::min(std::max(m_state.variable, -20.0f), 20.0f);
    }

    Float variable() const {
        return m_state.variable;
    }

private:
    struct State {
        int iter = 0;
        Float firstMoment = 0;
        Float secondMoment = 0;
        Float variable = 0;

        Float batchAccumulation = 0;
        Float batchGradient = 0;
    } m_state;

    struct Hyperparameters {
        Float learningRate;
        int batchSize;
        Float epsilon;
        Float beta1;
        Float beta2;
    } m_hparams;
};

enum class ESampleCombination {
    EDiscard,
    EDiscardWithAutomaticBudget,
    EInverseVariance,
};

enum class ESampleAllocSeq {
    EDouble,
    EUniform,
    EHalfdouble,
};

enum class EBsdfSamplingFractionLoss {
    ENone,
    EKL,
    EVariance,
};

enum class ESpatialFilter {
    ENearest,
    EStochasticBox,
    EBox,
};

enum class EDirectionalFilter {
    ENearest,
    EBox,
};





class QuadTreeNode {
public:
    QuadTreeNode() {
        m_children = {};
        for (size_t i = 0; i < m_sum.size(); ++i) {
            m_sum[i].store(0, std::memory_order_relaxed);
        }
    }

    void setSum(int index, Float val) {
        m_sum[index].store(val, std::memory_order_relaxed);
    }

    Float sum(int index) const {
        return m_sum[index].load(std::memory_order_relaxed);
    }

    void copyFrom(const QuadTreeNode& arg) {
        for (int i = 0; i < 4; ++i) {
            setSum(i, arg.sum(i));
            m_children[i] = arg.m_children[i];
        }
    }

    QuadTreeNode(const QuadTreeNode& arg) {
        copyFrom(arg);
    }

    QuadTreeNode& operator=(const QuadTreeNode& arg) {
        copyFrom(arg);
        return *this;
    }

    void setChild(int idx, uint16_t val) {
        m_children[idx] = val;
    }

    uint16_t child(int idx) const {
        return m_children[idx];
    }

    void setSum(Float val) {
        for (int i = 0; i < 4; ++i) {
            setSum(i, val);
        }
    }

    int childIndex(Point2& p) const {
        int res = 0;
        for (int i = 0; i < Point2::dim; ++i) {
            if (p[i] < 0.5f) {
                p[i] *= 2;
            } else {
                p[i] = (p[i] - 0.5f) * 2;
                res |= 1 << i;
            }
        }

        return res;
    }

    // Evaluates the directional irradiance *sum density* (i.e. sum / area) at a given location p.
    // To obtain radiance, the sum density (result of this function) must be divided
    // by the total statistical weight of the estimates that were summed up.
    Float eval(Point2& p, const std::vector<QuadTreeNode>& nodes) const {
        SAssert(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
        const int index = childIndex(p);
        if (isLeaf(index)) {
            return 4 * sum(index);
        } else {
            return 4 * nodes[child(index)].eval(p, nodes);
        }
    }

    Float pdf(Point2& p, const std::vector<QuadTreeNode>& nodes) const {
        SAssert(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
        const int index = childIndex(p);
        if (!(sum(index) > 0)) {
            return 0;
        }

        const Float factor = 4 * sum(index) / (sum(0) + sum(1) + sum(2) + sum(3));
        if (isLeaf(index)) {
            return factor;
        } else {
            return factor * nodes[child(index)].pdf(p, nodes);
        }
    }

    int depthAt(Point2& p, const std::vector<QuadTreeNode>& nodes) const {
        SAssert(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
        const int index = childIndex(p);
        if (isLeaf(index)) {
            return 1;
        } else {
            return 1 + nodes[child(index)].depthAt(p, nodes);
        }
    }


    Point2 sample(Sampler* sampler, const std::vector<QuadTreeNode>& nodes) const {
        int index = 0;

        Float topLeft = sum(0);
        Float topRight = sum(1);
        Float partial = topLeft + sum(2);
        Float total = partial + topRight + sum(3);

        // Should only happen when there are numerical instabilities.
        if (!(total > 0.0f)) {
            return sampler->next2D();
        }

        Float boundary = partial / total;
        Point2 origin = Point2{0.0f, 0.0f};

        Float sample = sampler->next1D();

        if (sample < boundary) {
            SAssert(partial > 0);
            sample /= boundary;
            boundary = topLeft / partial;
        } else {
            partial = total - partial;
            SAssert(partial > 0);
            origin.x = 0.5f;
            sample = (sample - boundary) / (1.0f - boundary);
            boundary = topRight / partial;
            index |= 1 << 0;
        }

        if (sample < boundary) {
            sample /= boundary;
        } else {
            origin.y = 0.5f;
            sample = (sample - boundary) / (1.0f - boundary);
            index |= 1 << 1;
        }

        if (isLeaf(index)) {
            return origin + 0.5f * sampler->next2D();
        } else {
            return origin + 0.5f * nodes[child(index)].sample(sampler, nodes);
        }
    }

    void record(Point2& p, Float irradiance, std::vector<QuadTreeNode>& nodes) {
        SAssert(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
        int index = childIndex(p);

        if (isLeaf(index)) {
            addToAtomicFloat(m_sum[index], irradiance);
        } else {
            nodes[child(index)].record(p, irradiance, nodes);
        }
    }

    Float computeOverlappingArea(const Point2& min1, const Point2& max1, const Point2& min2, const Point2& max2) {
        Float lengths[2];
        for (int i = 0; i < 2; ++i) {
            lengths[i] = std::max(std::min(max1[i], max2[i]) - std::max(min1[i], min2[i]), 0.0f);
        }
        return lengths[0] * lengths[1];
    }

    void record(const Point2& origin, Float size, Point2 nodeOrigin, Float nodeSize, Float value, std::vector<QuadTreeNode>& nodes) {
        Float childSize = nodeSize / 2;
        for (int i = 0; i < 4; ++i) {
            Point2 childOrigin = nodeOrigin;
            if (i & 1) { childOrigin[0] += childSize; }
            if (i & 2) { childOrigin[1] += childSize; }

            Float w = computeOverlappingArea(origin, origin + Point2(size), childOrigin, childOrigin + Point2(childSize));
            if (w > 0.0f) {
                if (isLeaf(i)) {
                    addToAtomicFloat(m_sum[i], value * w);
                } else {
                    nodes[child(i)].record(origin, size, childOrigin, childSize, value, nodes);
                }
            }
        }
    }

    bool isLeaf(int index) const {
        return child(index) == 0;
    }

    // Ensure that each quadtree node's sum of irradiance estimates
    // equals that of all its children.
    void build(std::vector<QuadTreeNode>& nodes) {
        for (int i = 0; i < 4; ++i) {
            // During sampling, all irradiance estimates are accumulated in
            // the leaves, so the leaves are built by definition.
            if (isLeaf(i)) {
                continue;
            }

            QuadTreeNode& c = nodes[child(i)];

            // Recursively build each child such that their sum becomes valid...
            c.build(nodes);

            // ...then sum up the children's sums.
            Float sum = 0;
            for (int j = 0; j < 4; ++j) {
                sum += c.sum(j);
            }
            setSum(i, sum);
        }
    }

private:
    std::array<std::atomic<Float>, 4> m_sum;
    std::array<uint16_t, 4> m_children;
};


template<int N>
struct QuadTreeNodeN {
    std::array<std::array<std::atomic<Float>, N>, 4> m_sum; // child[0][primal,adpos,adneg,pdf], child1[], child2[], child3[]
    std::array<uint16_t, 4> m_children;

    QuadTreeNodeN() {
        m_children = {};
        for (size_t i = 0; i < m_sum.size(); ++i) {
            for (int j = 0; j < N; ++j) {
                m_sum[i][j].store(0, std::memory_order_relaxed);
            }
        }
    }
    QuadTreeNodeN(const QuadTreeNodeN& arg) {
        copyFrom(arg);
    }

    QuadTreeNodeN& operator=(const QuadTreeNodeN& arg) {
        copyFrom(arg);
        return *this;
    }

    void copyFrom(const QuadTreeNodeN& arg) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < N; ++j) {
                setSum({ i,j }, arg.sum({ i,j }));
            }
            m_children[i] = arg.m_children[i];
        }
    }

    void setSum(Point2i index, Float val) {
        m_sum[index.x][index.y].store(val, std::memory_order_relaxed);
    }

    Float sum(Point2i index) const {
        return m_sum[index.x][index.y].load(std::memory_order_relaxed);
    }


    void setChild(int idx, uint16_t val) {
        m_children[idx] = val;
    }

    uint16_t child(int idx) const {
        return m_children[idx];
    }

    void setSum(Float val) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < N; ++j) {
                setSum({i, j}, val);
            }
        }
    }

    int childIndex(Point2& p) const {
        int res = 0;
        for (int i = 0; i < Point2::dim; ++i) {
            if (p[i] < 0.5f) {
                p[i] *= 2;
            }
            else {
                p[i] = (p[i] - 0.5f) * 2;
                res |= 1 << i;
            }
        }

        return res;
    }

    bool isLeaf(int index) const {
        return child(index) == 0;
    }
};

using QuadTreeNode4 = QuadTreeNodeN<4>;

struct Atomic {
    Atomic() {
        sum.store(0, std::memory_order_relaxed);
        statisticalWeight.store(0, std::memory_order_relaxed);
    }

    Atomic(const Atomic& arg) {
        *this = arg;
    }

    Atomic& operator=(const Atomic& arg) {
        sum.store(arg.sum.load(std::memory_order_relaxed), std::memory_order_relaxed);
        statisticalWeight.store(arg.statisticalWeight.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }

    std::atomic<Float> sum;
    std::atomic<Float> statisticalWeight;

};

class DTree {
public:
    DTree() {
        m_atomic.sum.store(0, std::memory_order_relaxed);
        m_maxDepth = 0;
        m_nodes.emplace_back();
        m_nodes.front().setSum(0.0f);
    }

    const QuadTreeNode& node(size_t i) const {
        return m_nodes[i];
    }

    Float mean() const {
        if (m_atomic.statisticalWeight == 0) {
            return 0;
        }
        const Float factor = 1 / (M_PI * 4 * m_atomic.statisticalWeight);
        return factor * m_atomic.sum;
    }

    void recordIrradiance(Point2 p, Float irradiance, Float statisticalWeight, EDirectionalFilter directionalFilter) {
        if (std::isfinite(statisticalWeight) && statisticalWeight > 0) {
            addToAtomicFloat(m_atomic.statisticalWeight, statisticalWeight);

            if (std::isfinite(irradiance) && irradiance > 0) {
                if (directionalFilter == EDirectionalFilter::ENearest) {
                    m_nodes[0].record(p, irradiance * statisticalWeight, m_nodes);
                } else {
                    int depth = depthAt(p);
                    Float size = std::pow(0.5f, depth);

                    Point2 origin = p;
                    origin.x -= size / 2;
                    origin.y -= size / 2;
                    m_nodes[0].record(origin, size, Point2(0.0f), 1.0f, irradiance * statisticalWeight / (size * size), m_nodes);
                }
            }
        }
    }

    Float pdf(Point2 p) const {
        if (!(mean() > 0)) {
            return 1 / (4 * M_PI);
        }

        return m_nodes[0].pdf(p, m_nodes) / (4 * M_PI);
    }

    Float eval(Point2 p) const {
        if (!(mean() > 0)) {
            return 0;
        }

        return m_nodes[0].eval(p, m_nodes) / statisticalWeight();
    }

    Float eval_raw(Point2 p) const {
        if (!(mean() > 0)) {
            return 0;
        }

        return m_nodes[0].eval(p, m_nodes);
    }

    int depthAt(Point2 p) const {
        return m_nodes[0].depthAt(p, m_nodes);
    }

    int depth() const {
        return m_maxDepth;
    }

    Point2 sample(Sampler* sampler) const {
        if (!(mean() > 0)) {
            return sampler->next2D();
        }

        Point2 res = m_nodes[0].sample(sampler, m_nodes);

        res.x = math::clamp(res.x, 0.0f, 1.0f);
        res.y = math::clamp(res.y, 0.0f, 1.0f);

        return res;
    }

    size_t numNodes() const {
        return m_nodes.size();
    }

    Float statisticalWeight() const {
        return m_atomic.statisticalWeight;
    }

    void setStatisticalWeight(Float statisticalWeight) {
        m_atomic.statisticalWeight = statisticalWeight;
    }

    void scale(float x) {
        for (auto& node : m_nodes) {
            for (int i = 0; i < 4; i ++ ) {
                node.setSum(i, node.sum(i) * x);
            }
        }
        maintain_sum();
    }

    void reset(const DTree& previousDTree, int newMaxDepth, Float subdivisionThreshold, bool refine_only = false, bool clean = true) {

        m_atomic = Atomic{};
        m_maxDepth = 0;
        m_nodes.clear();
        m_nodes.emplace_back();

        struct StackNode {
            size_t nodeIndex;
            size_t otherNodeIndex;
            const DTree* otherDTree;
            int depth;
        };

        std::stack<StackNode> nodeIndices;
        nodeIndices.push({0, 0, &previousDTree, 1});

        const Float total = previousDTree.m_atomic.sum;

        // Create the topology of the new DTree to be the refined version
        // of the previous DTree. Subdivision is recursive if enough energy is ther

        while (!nodeIndices.empty()) {
            statsResetBFSCall ++;
            StackNode sNode = nodeIndices.top();
            nodeIndices.pop();

            m_maxDepth = std::max(m_maxDepth, sNode.depth);

            for (int i = 0; i < 4; ++i) {
                const QuadTreeNode& otherNode = sNode.otherDTree->m_nodes[sNode.otherNodeIndex];
                const Float fraction = total > 0 ? (otherNode.sum(i) / total) : std::pow(0.25f, sNode.depth);
                SAssert(fraction <= 1.0f + Epsilon);

                // ? What if the number of sample is exactly zero ?
                if ((sNode.depth < newMaxDepth && fraction > subdivisionThreshold) || (refine_only && !otherNode.isLeaf(i))) {
                    if (!otherNode.isLeaf(i)) {
                        SAssert(sNode.otherDTree == &previousDTree);
                        nodeIndices.push({m_nodes.size(), otherNode.child(i), &previousDTree, sNode.depth + 1});
                    } else {
                        nodeIndices.push({m_nodes.size(), m_nodes.size(), this, sNode.depth + 1});
                    }

                    m_nodes[sNode.nodeIndex].setChild(i, static_cast<uint16_t>(m_nodes.size()));
                    m_nodes.emplace_back();
                    m_nodes.back().setSum(otherNode.sum(i) / 4);

                    if (m_nodes.size() > std::numeric_limits<uint16_t>::max()) {
                        SLog(EWarn, "DTreeWrapper hit maximum children count.");
                        nodeIndices = std::stack<StackNode>();
                        break;
                    }
                }
            }
        }

        // Uncomment once memory becomes an issue.
        //m_nodes.shrink_to_fit();

        for (auto& node : m_nodes) {
            node.setSum(0);
        }
        maintain_sum();
    }

    void reset2(const DTree& previousDTree, const DTree& previousDTree2, int maxDepth) {
        // Get the "max" structure of pDTree and pDTree2

        m_atomic = Atomic{};
        m_maxDepth = 0;
        m_nodes.clear();
        m_nodes.emplace_back();

        struct StackNode {
            size_t nodeIndex;
            size_t otherNodeIndex;
            size_t otherNodeIndex2;
            const DTree* otherDTree;
            const DTree* otherDTree2;
            int depth;
        };

        std::stack<StackNode> nodeIndices;
        nodeIndices.push({0, 0, 0, &previousDTree, &previousDTree2, 1});

        while (!nodeIndices.empty()) {
            statsResetBFSCall ++;
            StackNode sNode = nodeIndices.top();
            nodeIndices.pop();

            m_maxDepth = std::max(m_maxDepth, sNode.depth);

            for (int i = 0; i < 4; ++i) {
                const QuadTreeNode& otherNode = sNode.otherDTree->m_nodes[sNode.otherNodeIndex];
                const QuadTreeNode& otherNode2 = sNode.otherDTree2->m_nodes[sNode.otherNodeIndex2];

                // ? What if the number of sample is exactly zero ?
                if (!(otherNode.isLeaf(i) && otherNode2.isLeaf(i))) {
                    StackNode newStackNode = {m_nodes.size(), m_nodes.size(), m_nodes.size(), this, this, sNode.depth + 1};

                    if (!otherNode.isLeaf(i)) {
                        newStackNode.otherNodeIndex = otherNode.child(i);
                        newStackNode.otherDTree = &previousDTree;
                    }
                    if (!otherNode2.isLeaf(i)) {
                        newStackNode.otherNodeIndex2 = otherNode2.child(i);
                        newStackNode.otherDTree2 = &previousDTree2;
                    }

                    nodeIndices.push(newStackNode);

                    m_nodes[sNode.nodeIndex].setChild(i, static_cast<uint16_t>(m_nodes.size()));
                    m_nodes.emplace_back();

                    if (m_nodes.size() > std::numeric_limits<uint16_t>::max()) {
                        SLog(EWarn, "DTreeWrapper hit maximum children count.");
                        nodeIndices = std::stack<StackNode>();
                        break;
                    }
                }
            }
        }

        // Uncomment once memory becomes an issue.
        //m_nodes.shrink_to_fit();

        for (auto& node : m_nodes) {
            node.setSum(0);
        }
        maintain_sum();
    }

    size_t approxMemoryFootprint() const {
        return m_nodes.capacity() * sizeof(QuadTreeNode) + sizeof(*this);
    }

    void build() {
        auto& root = m_nodes[0];

        // Build the quadtree recursively, starting from its root.
        root.build(m_nodes);

        // Ensure that the overall sum of irradiance estimates equals
        // the sum of irradiance estimates found in the quadtree.
        Float sum = 0;
        for (int i = 0; i < 4; ++i) {
            sum += root.sum(i);
        }
        m_atomic.sum.store(sum);
    }

    void maintain_sum()
    {
        auto& root = m_nodes[0];
        Float sum = 0;
        for (int i = 0; i < 4; ++i)
        {
            sum += root.sum(i);
        }
        m_atomic.sum.store(sum);
    }

    void superfuse_pushdown(int p, int i, Float val)
    {
        statsSuperfusePushdownCall++;
        if (!m_nodes[p].isLeaf(i))
        {
            int q = m_nodes[p].child(i);
            val /= 4;
            for (int j = 0; j < 4; j++)
            {
                Float x = m_nodes[q].sum(j);
                Float y = val;
                m_nodes[q].setSum(j, x + y);
                if (!m_nodes[q].isLeaf(j))
                {
                    superfuse_pushdown(q, j, y);
                }
            }
        }
    }

    void superfuse_dfs(const DTree &src, int p, int q, float alpha)
    {
        statsSuperfuseDFSCall++;
        for (int i = 0; i < 4; i++)
        {
            Float x = m_nodes[p].sum(i);
            Float y = src.m_nodes[q].sum(i) * alpha;
            m_nodes[p].setSum(i, x + y);
            if (!m_nodes[p].isLeaf(i))
            {
                if (!src.m_nodes[q].isLeaf(i))
                {
                    superfuse_dfs(src, m_nodes[p].child(i), src.m_nodes[q].child(i), alpha);
                }
                else
                {
                    superfuse_pushdown(p, i, y);
                }
            }
        }
    }

    // Add a DTree 'src' to this
    void superfuse(const DTree &src, float alpha = 1)
    {
        DTree old = *this;
        superfuse_dfs(src, 0, 0, alpha);
        if (alpha > 0) {
            setStatisticalWeight(statisticalWeight() + src.statisticalWeight() * alpha);
        }
        maintain_sum();

    }

    void add(float value = 0) {
        for (int i = 0; i < 4; i++) {
            Float x = m_nodes[0].sum(i);
            Float y = value / 4;
            m_nodes[0].setSum(i, x + y);
            superfuse_pushdown(0, i, value / 4);
        }
        maintain_sum();
    }

    void normalize_dfs(int p, Float sum)
    {
        for (int i = 0; i < 4; i++)
        {
            Float x = m_nodes[p].sum(i);
            m_nodes[p].setSum(i, x / sum);
            if (!m_nodes[p].isLeaf(i))
            {
                normalize_dfs(m_nodes[p].child(i), sum);
            }
        }
    }

    void normalize()
    {
        Float sum = m_nodes[0].sum(0) + m_nodes[0].sum(1) + m_nodes[0].sum(2) + m_nodes[0].sum(3);
        if (sum < 1e-6) {
            add(1e-4);
        }
        sum = m_nodes[0].sum(0) + m_nodes[0].sum(1) + m_nodes[0].sum(2) + m_nodes[0].sum(3);
        normalize_dfs(0, sum);
        maintain_sum();
    }

    void selfcheck_dfs(int p, Float sum) const
    {
        Float t = 0;
        for (int i = 0; i < 4 ; i++) 
        {
            if (m_nodes[p].sum(i) < 0) {
                std::cout << "self check failed negative " << p << " " << i << " " << m_nodes[p].sum(i) << std::endl;
            }
            if (!m_nodes[p].isLeaf(i))
            {
                selfcheck_dfs(m_nodes[p].child(i), m_nodes[p].sum(i));
            }
            t += m_nodes[p].sum(i);
        }
        Float delta = abs(t - sum);
        Float mx = std::max(t, sum);
        if (delta > 0.01 && delta > 0.001 * mx)
        {
            std::cout << "self check failed sum t=" << t << " sum=" << sum << " p=" << p << std::endl;
        }
    }

    void selfcheck() const
    {
        selfcheck_dfs(0, m_atomic.sum);
    }

    void my_vis_dfs(int i, Point2f pos, Vector2f siz) const {
        std::stringstream ss;
        ss << "  dnode " << i << " ";
        ss << "pos=" << pos.toString() << " ";
        ss << "size=" << siz.toString() << " ";
        ss << "ch0=" << m_nodes[i].child(0) << " ";
        ss << "ch1=" << m_nodes[i].child(1) << " ";
        ss << "ch2=" << m_nodes[i].child(2) << " ";
        ss << "ch3=" << m_nodes[i].child(3) << " ";
        ss << "v0=" << m_nodes[i].sum(0) << " ";
        ss << "v1=" << m_nodes[i].sum(1) << " ";
        ss << "v2=" << m_nodes[i].sum(2) << " ";
        ss << "v3=" << m_nodes[i].sum(3) << " ";
        std::cout << ss.str() << std::endl;

        siz /= 2;
        if (!m_nodes[i].isLeaf(0)) my_vis_dfs(m_nodes[i].child(0), pos, siz);
        pos[1] += siz[1];
        if (!m_nodes[i].isLeaf(1)) my_vis_dfs(m_nodes[i].child(1), pos, siz);
        pos[0] += siz[0];
        pos[1] -= siz[1];
        if (!m_nodes[i].isLeaf(2)) my_vis_dfs(m_nodes[i].child(2), pos, siz);
        pos[1] += siz[1];
        if (!m_nodes[i].isLeaf(3)) my_vis_dfs(m_nodes[i].child(3), pos, siz);
    }

    void my_vis() const {
        my_vis_dfs(0, Point2f(0.0f), Vector2f(1.0f));
    }

    std::string my_export_dfs(int i, Point2f pos, Vector2f siz, OFSTREAM& ofs) const {
        
        SSTREAM ss;
        ss << i << " ";
        ss << m_nodes[i].child(0) << " ";
        ss << m_nodes[i].child(1) << " ";
        ss << m_nodes[i].child(2) << " ";
        ss << m_nodes[i].child(3) << " ";
        ss << m_nodes[i].sum(0) << " ";
        ss << m_nodes[i].sum(1) << " ";
        ss << m_nodes[i].sum(2) << " ";
        ss << m_nodes[i].sum(3);
        ss << std::endl;
       
        std::string parallel_results[4] = {"", "", "", ""};
        
        Point2f child_p_0(pos[0], pos[1]);
        Point2f child_p_1(pos[0], pos[1] + siz[1]);
        Point2f child_p_2(pos[0] + siz[0], pos[1]);
        Point2f child_p_3(pos[0] + siz[0], pos[1] + siz[1]);
        Point2f child_p[4] = { child_p_0, child_p_1, child_p_2, child_p_3};

        siz /= 2;

#pragma omp parallel for
        for (int j = 0; j < 4; j++)
        { 
            if (!m_nodes[i].isLeaf(j)) parallel_results[j] = my_export_dfs(m_nodes[i].child(j), child_p[j], siz, ofs);
        };

        for (size_t j = 0; j < 4; j++)
        {
            ss << parallel_results[j];
        }
        std::string result = ss.str();
        return result;

       
    }

    void my_export(OFSTREAM& ofs) const {
        ofs << m_nodes.size() << " " << m_atomic.statisticalWeight << std::endl;
        ofs << my_export_dfs(0, Point2f(0.0f), Vector2f(1.0f), ofs).c_str();
        ofs << "-1" << std::endl;
    }


    void my_import(std::ifstream& ifs) {
        int n;
        ifs >> n;
        m_nodes.clear();
        m_nodes.resize(n);

        int p, p0, p1, p2, p3;
        float v0, v1, v2, v3;
        while (true) {
            ifs >> p;
            if (p == -1) {
                break;
            }

            ifs >> p0 >> p1 >> p2 >> p3;
            ifs >> v0 >> v1 >> v2 >> v3;

            m_nodes[p].setChild(0, p0);
            m_nodes[p].setChild(1, p1);
            m_nodes[p].setChild(2, p2);
            m_nodes[p].setChild(3, p3);

            m_nodes[p].setSum(0, v0);
            m_nodes[p].setSum(1, v1);
            m_nodes[p].setSum(2, v2);
            m_nodes[p].setSum(3, v3);
        }
    }

    float sum() const {
        return m_atomic.sum;
    }

private:
    std::vector<QuadTreeNode> m_nodes;

    Atomic m_atomic;

    int m_maxDepth;
};



int g_iter = 0;
int g_passesThisIter = 0;
int g_passesDone = 0;


struct DTreeRecord {
    Vector d;
    Float radiance, product;
    Float woPdf, bsdfPdf, dTreePdf;
    Float statisticalWeight;
    bool isDelta;
    Point3f p;
};

struct DTreeWrapper {
public:
    Point3f any_position;

    DTreeWrapper() {
    }

    void record(const DTreeRecord& rec, EDirectionalFilter directionalFilter, EBsdfSamplingFractionLoss bsdfSamplingFractionLoss) {
        any_position = rec.p;
        if (!rec.isDelta) {
            Float irradiance = rec.radiance / rec.woPdf;
            building.recordIrradiance(dirToCanonical(rec.d), irradiance, rec.statisticalWeight, directionalFilter);
        }

        if (bsdfSamplingFractionLoss != EBsdfSamplingFractionLoss::ENone && rec.product > 0) {
            optimizeBsdfSamplingFraction(rec, bsdfSamplingFractionLoss == EBsdfSamplingFractionLoss::EKL ? 1.0f : 2.0f);
        }
    }

    static Vector canonicalToDir(Point2 p) {
        const Float cosTheta = 2 * p.x - 1;
        const Float phi = 2 * M_PI * p.y;

        const Float sinTheta = sqrt(1 - cosTheta * cosTheta);
        Float sinPhi, cosPhi;
        math::sincos(phi, &sinPhi, &cosPhi);

        return {sinTheta * cosPhi, sinTheta * sinPhi, cosTheta};
    }

    static Point2 dirToCanonical(const Vector& d) {
        if (!std::isfinite(d.x) || !std::isfinite(d.y) || !std::isfinite(d.z)) {
            return {0, 0};
        }

        const Float cosTheta = std::min(std::max(d.z, -1.0f), 1.0f);
        Float phi = std::atan2(d.y, d.x);
        while (phi < 0)
            phi += 2.0 * M_PI;

        return {(cosTheta + 1) / 2, phi / (2 * M_PI)};
    }

    void build() {
        HDTimer timer_build;
        building.build();
        sampling = building;
        statsSdtreeBuild += timer_build.value();
    }

    void reset(int maxDepth, Float subdivisionThreshold) {
        HDTimer timer_reset;
        building.reset(sampling, maxDepth, subdivisionThreshold, false, true);
        if (g_stepper) {
            building.superfuse(sampling, global_decay_rate);
        }
        statsSdtreeReset += timer_reset.value();
    }

    Vector sample(Sampler* sampler) const {
        return canonicalToDir(sampling.sample(sampler));
    }

    Float pdf(const Vector& dir) const {
        return sampling.pdf(dirToCanonical(dir));
    }

    Float eval(const Vector& dir) const {
        return building.eval(dirToCanonical(dir));
    }

    Float pdf_fused(const Vector& dir) const {
        return fused.pdf(dirToCanonical(dir));
    }

    Float diff(const DTreeWrapper& other) const {
        return 0.0f;
    }

    int depth() const {
        return sampling.depth();
    }

    size_t numNodes() const {
        return sampling.numNodes();
    }

    Float meanRadiance() const {
        return sampling.mean();
    }

    Float statisticalWeight() const {
        return sampling.statisticalWeight();
    }

    Float statisticalWeightBuilding() const {
        return building.statisticalWeight();
    }

    void setStatisticalWeightBuilding(Float statisticalWeight) {
        building.setStatisticalWeight(statisticalWeight);
    }

    void setStatisticalWeightSampling(Float statisticalWeight) {
        sampling.setStatisticalWeight(statisticalWeight);
    }

    size_t approxMemoryFootprint() const {
        return building.approxMemoryFootprint() + sampling.approxMemoryFootprint();
    }

    inline Float bsdfSamplingFraction(Float variable) const {
        return logistic(variable);
    }

    inline Float dBsdfSamplingFraction_dVariable(Float variable) const {
        Float fraction = bsdfSamplingFraction(variable);
        return fraction * (1 - fraction);
    }

    inline Float bsdfSamplingFraction() const {
        return bsdfSamplingFraction(bsdfSamplingFractionOptimizer.variable());
    }

    void optimizeBsdfSamplingFraction(const DTreeRecord& rec, Float ratioPower) {
        m_lock.lock();

        // GRADIENT COMPUTATION
        Float variable = bsdfSamplingFractionOptimizer.variable();
        Float samplingFraction = bsdfSamplingFraction(variable);

        // Loss gradient w.r.t. sampling fraction
        Float mixPdf = samplingFraction * rec.bsdfPdf + (1 - samplingFraction) * rec.dTreePdf;
        Float ratio = std::pow(rec.product / mixPdf, ratioPower);
        Float dLoss_dSamplingFraction = -ratio / rec.woPdf * (rec.bsdfPdf - rec.dTreePdf);

        // Chain rule to get loss gradient w.r.t. trainable variable
        Float dLoss_dVariable = dLoss_dSamplingFraction * dBsdfSamplingFraction_dVariable(variable);

        // We want some regularization such that our parameter does not become too big.
        // We use l2 regularization, resulting in the following linear gradient.
        Float l2RegGradient = 0.01f * variable;

        Float lossGradient = l2RegGradient + dLoss_dVariable;

        // ADAM GRADIENT DESCENT
        bsdfSamplingFractionOptimizer.append(lossGradient, rec.statisticalWeight);

        m_lock.unlock();
    }

    void dump(BlobWriter& blob, const Point& p, const Vector& size) const {
        blob
            << (float)p.x << (float)p.y << (float)p.z
            << (float)size.x << (float)size.y << (float)size.z
            << (float)sampling.mean() << (uint64_t)sampling.statisticalWeight() << (uint64_t)sampling.numNodes();

        for (size_t i = 0; i < sampling.numNodes(); ++i) {
            const auto& node = sampling.node(i);
            for (int j = 0; j < 4; ++j) {
                blob << (float)node.sum(j) << (uint16_t)node.child(j);
            }
        }
    }

    void scale(float x) {
        building.scale(x);
        sampling.scale(x);
    }

    void my_export(OFSTREAM& ofs) const {
        // ofs << sampling.mean() << " " << sampling.statisticalWeight() << " " << sampling.numNodes() << std::endl;
        sampling.my_export(ofs);
    }

    void my_import(std::ifstream& ifs) {
        sampling.my_import(ifs);
    }

    void my_vis() const {
        std::cout << "==== building ";
        building.selfcheck();
        building.my_vis();
        std::cout << std::endl;
        std::cout << std::endl;

        std::cout << "==== sampling ";
        sampling.selfcheck();
        sampling.my_vis();
        std::cout << std::endl;
        std::cout << std::endl;

        std::cout << "==== fused ";
        fused.selfcheck();
        fused.my_vis();
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }

    void selfcheck() const {
        std::cout << "DTree selfcheck building" << std::endl;
        building.selfcheck();
        std::cout << "DTree selfcheck sampling" << std::endl;
        sampling.selfcheck();
        std::cout << "DTree selfcheck fused" << std::endl;
        fused.selfcheck();
    }


    DTree building;
    DTree sampling;
    DTree sampling_backup;
    DTree fused;

    AdamOptimizer bsdfSamplingFractionOptimizer{0.01f};

    class SpinLock {
    public:
        SpinLock() {
            m_mutex.clear(std::memory_order_release);
        }

        SpinLock(const SpinLock& other) { m_mutex.clear(std::memory_order_release); }
        SpinLock& operator=(const SpinLock& other) { return *this; }

        void lock() {
            while (m_mutex.test_and_set(std::memory_order_acquire)) { }
        }

        void unlock() {
            m_mutex.clear(std::memory_order_release);
        }
    private:
        std::atomic_flag m_mutex;
    } m_lock;
};


enum class EDTreeType {
    EPrimal = 0,
    EAdPos,
    EAdNeg,

    EdLf_Pos,
    EdLf_Neg,
    ELdf_Pos,
    ELdf_Neg,
    ELf,

    EDTreeType_Count
};


struct STreeNode {
    STreeNode() {
        children = {};
        isLeaf = true;
        axis = 0;
    }

    int childIndex(Point& p) const {
        if (p[axis] < 0.5f) {
            p[axis] *= 2;
            return 0;
        } else {
            p[axis] = (p[axis] - 0.5f) * 2;
            return 1;
        }
    }

    int nodeIndex(Point& p) const {
        return children[childIndex(p)];
    }

    DTreeWrapper* dTreeWrapper(Point& p, Vector& size, std::vector<STreeNode>& nodes, EDTreeType dtree_type = EDTreeType::EPrimal) {
        SAssert(p[axis] >= 0 && p[axis] <= 1);
        if (isLeaf) {
            return &dTree;
        } else {
            size[axis] /= 2;
            return nodes[nodeIndex(p)].dTreeWrapper(p, size, nodes);
        }
    }

    const DTreeWrapper* dTreeWrapper(EDTreeType dtree_type = EDTreeType::EPrimal) const {
        return &dTree;
    }

    DTreeWrapper* dTreeWrapperNonconst(EDTreeType dtree_type = EDTreeType::EPrimal) {
        return &dTree;
    }

    int depth(Point& p, const std::vector<STreeNode>& nodes) const {
        SAssert(p[axis] >= 0 && p[axis] <= 1);
        if (isLeaf) {
            return 1;
        } else {
            return 1 + nodes[nodeIndex(p)].depth(p, nodes);
        }
    }

    int depth(const std::vector<STreeNode>& nodes) const {
        int result = 1;

        if (!isLeaf) {
            for (auto c : children) {
                result = std::max(result, 1 + nodes[c].depth(nodes));
            }
        }

        return result;
    }

    void forEachLeaf(
        std::function<void(const DTreeWrapper*, const Point&, const Vector&)> func,
        Point p, Vector size, const std::vector<STreeNode>& nodes) const {

        if (isLeaf) {
            func(&dTree, p, size);
        } else {
            size[axis] /= 2;
            for (int i = 0; i < 2; ++i) {
                Point childP = p;
                if (i == 1) {
                    childP[axis] += size[axis];
                }

                nodes[children[i]].forEachLeaf(func, childP, size, nodes);
            }
        }
    }

    Float computeOverlappingVolume(const Point& min1, const Point& max1, const Point& min2, const Point& max2) {
        Float lengths[3];
        for (int i = 0; i < 3; ++i) {
            lengths[i] = std::max(std::min(max1[i], max2[i]) - std::max(min1[i], min2[i]), 0.0f);
        }
        return lengths[0] * lengths[1] * lengths[2];
    }

    void record(const Point& min1, const Point& max1, Point min2, Vector size2, const DTreeRecord& rec,
        EDirectionalFilter directionalFilter, EBsdfSamplingFractionLoss bsdfSamplingFractionLoss,
        std::vector<STreeNode>& nodes, EDTreeType dtree_type = EDTreeType::EPrimal) {
        Float w = computeOverlappingVolume(min1, max1, min2, min2 + size2);
        if (w > 0) {
            if (isLeaf) {
                dTree.record({ rec.d, rec.radiance, rec.product, rec.woPdf, rec.bsdfPdf, rec.dTreePdf, rec.statisticalWeight * w, rec.isDelta, min2 }, directionalFilter, bsdfSamplingFractionLoss);
            } else {
                size2[axis] /= 2;
                for (int i = 0; i < 2; ++i) {
                    if (i & 1) {
                        min2[axis] += size2[axis];
                    }

                    nodes[children[i]].record(min1, max1, min2, size2, rec, directionalFilter, bsdfSamplingFractionLoss, nodes);
                }
            }
        }
    }

    int getNodeId(Point& p, const std::vector<STreeNode>& nodes, int cur, int dlim) const {
        SAssert(p[axis] >= 0 && p[axis] <= 1);
        if (isLeaf || dlim == 0) {
            return cur;
        }
        else {
            int ni = nodeIndex(p);
            return nodes[ni].getNodeId(p, nodes, ni, dlim - 1);
        }
    }

    bool isLeaf;
    DTreeWrapper dTree;
    int axis;
    std::array<uint32_t, 2> children;
};

template<int N, bool PRB_MIX_3> 
struct STreeNodeN {

    // possible combinations
    //              prb                     srb        individual / pos&neg / all
    static_assert((PRB_MIX_3 && N == 3) || (!PRB_MIX_3 && (N == 1 || N == 2 || N == 5)));

    std::array<DTreeWrapper, N> dtrees;

    static constexpr int QuadTreeNodeEntryNum = N + int(PRB_MIX_3);  // in the version PRB_MIX_3, there is a additional entry in QuadTreeNode to store mixed pdf
    mutable std::vector<QuadTreeNodeN<QuadTreeNodeEntryNum>> mixNodes; // merged dtree
    mutable Atomic m_atomic[N];
    // mutable so that we can export const tree...

    bool isLeaf;
    int axis;
    std::array<uint32_t, 2> children;

    // Dtree dfs export results, currently stored as std::string
    std::string dtree_export_results;

private:
    static constexpr int remap(EDTreeType type)
    {
        if constexpr (N == 5) {
            switch (type) {
                case EDTreeType::EdLf_Pos: return 0;
                case EDTreeType::EdLf_Neg: return 1;
                case EDTreeType::ELdf_Pos: return 2;
                case EDTreeType::ELdf_Neg: return 3;
                case EDTreeType::ELf: return 4;
            }
        }
        else if constexpr (N == 2) {
            switch (type) {
                case EDTreeType::EdLf_Pos:
                case EDTreeType::ELdf_Pos:
                    return 0;
                case EDTreeType::EdLf_Neg:
                case EDTreeType::ELdf_Neg:
                    return 1;
            }
        }
        return 0;
    }
public:

    static int dtreeTypeToIdx(EDTreeType type) {
        constexpr int indices[int(EDTreeType::EDTreeType_Count)]{
            0,  // EPrimal, nothing special
            1,  // EAdPos
            2,  // EAdNeg
            remap(EDTreeType::EdLf_Pos),
            remap(EDTreeType::EdLf_Neg),
            remap(EDTreeType::ELdf_Pos),
            remap(EDTreeType::ELdf_Neg),
            remap(EDTreeType::ELf)
        };
        return indices[int(type)];
    }

    // ==================================================================

    STreeNodeN() {
        children = {};
        isLeaf = true;
        axis = 0;
        for (int i = 0; i < N; ++i) dtrees[i] = {};

        mixNodes.clear();
        mixNodes.emplace_back();
        for (int i = 0; i < N; ++i) m_atomic[i].statisticalWeight = m_atomic[i].sum = 0;
    }

    int childIndex(Point& p) const {
        if (p[axis] < 0.5f) {
            p[axis] *= 2;
            return 0;
        }
        else {
            p[axis] = (p[axis] - 0.5f) * 2;
            return 1;
        }
    }

    int nodeIndex(Point& p) const {
        return children[childIndex(p)];
    }

    DTreeWrapper* dTreeWrapper(Point& p, Vector& size, std::vector<STreeNodeN>& nodes, EDTreeType dtree_type = EDTreeType::EPrimal) {
        // bug record
        if (p[axis] < 0 || p[axis] > 1) {
            std::cout << "dTreeWrapper error ==============================\n";
            std::cout << axis << ' ' << p.x  << ' ' << p.y << ' ' << p.z << '\n';
            // so far, err value is at most 1.0004
            p[axis] = math::clamp(p[axis], Float(0), Float(1));
        }
        SAssert(p[axis] >= 0 && p[axis] <= 1);
        if (isLeaf) {
            return &dtrees[dtreeTypeToIdx(dtree_type)];
        }
        else {
            size[axis] /= 2;
            return nodes[nodeIndex(p)].dTreeWrapper(p, size, nodes, dtree_type);
        }
    }

    const DTreeWrapper* dTreeWrapper(EDTreeType dtree_type = EDTreeType::EPrimal) const {
        return &dtrees[dtreeTypeToIdx(dtree_type)];
    }

    DTreeWrapper* dTreeWrapperNonconst(EDTreeType dtree_type = EDTreeType::EPrimal) {
        return &dtrees[dtreeTypeToIdx(dtree_type)];
    }

    int depth(Point& p, const std::vector<STreeNodeN>& nodes) const {
        SAssert(p[axis] >= 0 && p[axis] <= 1);
        if (isLeaf) {
            return 1;
        }
        else {
            return 1 + nodes[nodeIndex(p)].depth(p, nodes);
        }
    }

    int getNodeId(Point& p, const std::vector<STreeNodeN>& nodes, int cur, int dlim) const {
        SAssert(p[axis] >= 0 && p[axis] <= 1);
        if (isLeaf || dlim == 0) {
            return cur;
        }
        else {
            int ni = nodeIndex(p);
            return nodes[ni].getNodeId(p, nodes, ni, dlim - 1);
        }
    }

    int depth(const std::vector<STreeNodeN>& nodes) const {
        int result = 1;

        if (!isLeaf) {
            for (auto c : children) {
                result = std::max(result, 1 + nodes[c].depth(nodes));
            }
        }

        return result;
    }

    void forEachLeaf(
        std::function<void(const DTreeWrapper*, const Point&, const Vector&)> func,
        Point p, Vector size, const std::vector<STreeNodeN>& nodes) const {

        if (isLeaf) {
            for (const auto& dtree : dtrees)
                func(&dtree, p, size);
        }
        else {
            size[axis] /= 2;
            for (int i = 0; i < 2; ++i) {
                Point childP = p;
                if (i == 1) {
                    childP[axis] += size[axis];
                }

                nodes[children[i]].forEachLeaf(func, childP, size, nodes);
            }
        }
    }

    Float computeOverlappingVolume(const Point& min1, const Point& max1, const Point& min2, const Point& max2) {
        Float lengths[3];
        for (int i = 0; i < 3; ++i) {
            lengths[i] = std::max(std::min(max1[i], max2[i]) - std::max(min1[i], min2[i]), 0.0f);
        }
        return lengths[0] * lengths[1] * lengths[2];
    }

    void record(const Point& min1, const Point& max1, Point min2, Vector size2, const DTreeRecord& rec,
        EDirectionalFilter directionalFilter, EBsdfSamplingFractionLoss bsdfSamplingFractionLoss,
        std::vector<STreeNodeN>& nodes, EDTreeType dtree_type = EDTreeType::EPrimal) {
        Float w = computeOverlappingVolume(min1, max1, min2, min2 + size2);
        if (w > 0) {
            if (isLeaf) {
                dtrees[dtreeTypeToIdx(dtree_type)].record({ rec.d, rec.radiance, rec.product, rec.woPdf, rec.bsdfPdf, rec.dTreePdf, rec.statisticalWeight * w, rec.isDelta, min2 }, directionalFilter, bsdfSamplingFractionLoss);
            }
            else {
                size2[axis] /= 2;
                for (int i = 0; i < 2; ++i) {
                    if (i & 1) {
                        min2[axis] += size2[axis];
                    }

                    nodes[children[i]].record(min1, max1, min2, size2, rec, directionalFilter, bsdfSamplingFractionLoss, nodes, dtree_type);
                }
            }
        }
    }


    // merge 3 dtree
    void generate_merged_result() const {

        for (int i = 0; i < N; ++i) m_atomic[i] = Atomic{};
        // helper functions
        auto maintain_sum = [&]() {
            auto& root = mixNodes[0];
            for (int slot = 0; slot < N; ++slot) {
                Float sum = 0;
                for (int i = 0; i < 4; ++i) {
                   sum += root.sum({i, slot});
                }
                m_atomic[slot].sum.store(sum);
            }
        };
        auto setStatisticalWeight = [&](Float value, int slot) {
            m_atomic[slot].statisticalWeight = value;
        };
        auto statisticalWeight = [&](int slot) -> Float {
            return m_atomic[slot].statisticalWeight;
        };

        // get the 'max' structure
        // see DTree::reset2() for reference
        auto resetN = [&]() {
            mixNodes.clear();
            mixNodes.emplace_back();

            struct StackNode {
                size_t nodeIndex;
                int depth;
                std::array<const QuadTreeNode*, N> otherNodes;

                StackNode(size_t index, int d) : nodeIndex(index), depth(d) {
                    for (auto& node : otherNodes) node = nullptr;
                }
            };

            std::stack<StackNode> nodeIndices;
            nodeIndices.push(StackNode(0, 1));
            for (int i = 0; i < N; ++i) {
                nodeIndices.top().otherNodes[i] = &dtrees[i].sampling.node(0);
            }

            while (!nodeIndices.empty()) {
                StackNode sNode = nodeIndices.top();
                nodeIndices.pop();

                for (int i = 0; i < 4; ++i) {
                    bool cond = false;
                    for (auto node : sNode.otherNodes) {
                        cond |= node != nullptr && node->isLeaf(i) == false;
                    }

                    // ? What if the number of sample is exactly zero ?
                    if (cond) {
                        StackNode newStackNode(mixNodes.size(), sNode.depth + 1);

                        for (int j = 0; j < N; ++j) {
                            auto node = sNode.otherNodes[j];
                            if (node && node->isLeaf(i) == false) {
                                newStackNode.otherNodes[j] = &dtrees[j].sampling.node(node->child(i));
                            }
                        }

                        nodeIndices.push(newStackNode);

                        mixNodes[sNode.nodeIndex].setChild(i, static_cast<uint16_t>(mixNodes.size()));
                        mixNodes.emplace_back();

                        if (mixNodes.size() > std::numeric_limits<uint16_t>::max()) {
                            SLog(EWarn, "DTreeWrapper hit maximum children count.");
                            nodeIndices = std::stack<StackNode>();
                            break;
                        }
                    }
                }
            }

            // Uncomment once memory becomes an issue.
            //m_nodes.shrink_to_fit();

            for (auto& node : mixNodes) {
                node.setSum(0);
            }
            maintain_sum();
        };

        resetN();

        // then set the value
        // see DTree::superfuse() for reference

        auto superfuse_pushdown = [&](auto&& self, int p, int i, Float val, int slot) -> void
        {
            if (!mixNodes[p].isLeaf(i))
            {
                int q = mixNodes[p].child(i);
                val /= 4;
                for (int j = 0; j < 4; j++)
                {
                    Point2i idx{ j, slot };
                    Float x = mixNodes[q].sum(idx);
                    Float y = val;
                    mixNodes[q].setSum(idx, x + y);
                    if (!mixNodes[q].isLeaf(j))
                    {
                        self(self, q, j, y, slot);
                    }
                }
            }
        };

        auto superfuse_dfs = [&](auto&& self, const DTree& src, int p, int q, float alpha, int slot) -> void
        {
            for (int i = 0; i < 4; i++)
            {
                Point2i idx{ i, slot };
                Float x = mixNodes[p].sum(idx);
                Float y = src.node(q).sum(i) * alpha;
                mixNodes[p].setSum(idx, x + y);
                if (!mixNodes[p].isLeaf(i))
                {
                    if (!src.node(q).isLeaf(i))
                    {
                        self(self, src, mixNodes[p].child(i), src.node(q).child(i), alpha, slot);
                    }
                    else
                    {
                        superfuse_pushdown(superfuse_pushdown, p, i, y, slot);
                    }
                }
            }
        };

        auto superfuse = [&](const DTree& src, float alpha, int slot)
        {
            superfuse_dfs(superfuse_dfs, src, 0, 0, alpha, slot);
            if (alpha > 0) {
                setStatisticalWeight(statisticalWeight(slot) + src.statisticalWeight() * alpha, slot);
            }
            maintain_sum();
        };

        // add each value to its position
        for (int i = 0; i < N; ++i) {
            superfuse(dtrees[i].sampling, 1, i);
        }

        if constexpr (PRB_MIX_3 == true) {
            // fill the pdf
            Float invSum[3];
            for (int i = 0; i < 3; ++i) invSum[i] = 1.0 / (m_atomic[i].sum + 1e-18);
            for (auto& node : mixNodes) {
                for (int i = 0; i < 4; ++i) {
                    // pdf
                    float gw1 = 1 - global_guiding_mis_weight_ad;
                    float gw2 = global_guiding_mis_weight_ad * 0.5;
                    float val = node.sum({ i, 0 }) * invSum[0] * gw1 
                        + node.sum({ i, 1 }) * invSum[1] * gw2
                        + node.sum({ i, 2 }) * invSum[2] * gw2;
                    float weight = (m_atomic[0].sum > 0) * gw1 + (m_atomic[1].sum > 0) * gw2 + (m_atomic[2].sum > 0) * gw2;
                    if (weight == 0) 
                        weight = 1;
                    node.setSum({ i, 3 }, val / weight);
                }
            }
        }
        maintain_sum();
    }
    std::string my_export_dfs(int i, Point2f pos, Vector2f siz, OFSTREAM& ofs) const {
        
        SSTREAM ss;
        ss << i << " ";
        ss << (int)mixNodes[i].child(0) << " ";
        ss << (int)mixNodes[i].child(1) << " ";
        ss << (int)mixNodes[i].child(2) << " ";
        ss << (int)mixNodes[i].child(3) << " ";

        for (int j = 0; j < 4; ++j)
            for (int slot = 0; slot < QuadTreeNodeEntryNum; ++slot)
                ss << mixNodes[i].sum({ j, slot }) << " ";
        ss << std::endl;
       
        siz /= 2;

        Point2f child_p_0(pos[0], pos[1]);
        Point2f child_p_1(pos[0], pos[1] + siz[1]);
        Point2f child_p_2(pos[0] + siz[0], pos[1]);
        Point2f child_p_3(pos[0] + siz[0], pos[1] + siz[1]);
        Point2f child_p[4] = { child_p_0, child_p_1, child_p_2, child_p_3 };
        std::string parallel_results[4] = {"", "", "", ""};

//parallel dfs?
//#pragma omp parallel for
        for (int j = 0; j < 4; j++)
        {
            if (!mixNodes[i].isLeaf(j)) parallel_results[j] = my_export_dfs(mixNodes[i].child(j), child_p[j], siz, ofs);
        }

       
        for (size_t j = 0; j < 4; j++){
            ss << parallel_results[j];
        }
        std::string result = ss.str();

        return result;
    };

    void my_export_dtree(OFSTREAM& ofs){
        dtree_export_results = my_export_dfs(0, Point2f(0.0f), Vector2f(1.0f), ofs);
    }

    void my_export(OFSTREAM& ofs) const {

        // see DTree::my_export() for reference
       
        // note: pos&neg part dtree SW not strictly equal when splat by EStochasticBox
        if constexpr (PRB_MIX_3 == true) {
            //SAssert(std::abs(dtrees[1].sampling.statisticalWeight() - dtrees[2].sampling.statisticalWeight()) < 1);
            ofs << mixNodes.size() << ' ' 
                << dtrees[0].sampling.statisticalWeight() << ' '
                << (dtrees[1].sampling.statisticalWeight() + dtrees[2].sampling.statisticalWeight()) / 2 << std::endl;
                // note in before cv version, only output 'dtrees[2].sampling.statisticalWeight()'
        }
        else {
            static_assert(N == 5);
            //SAssert(std::abs(dtrees[0].sampling.statisticalWeight() - dtrees[1].sampling.statisticalWeight()) < 1);
            //SAssert(std::abs(dtrees[2].sampling.statisticalWeight() - dtrees[3].sampling.statisticalWeight()) < 1);
            ofs << mixNodes.size() << ' ' 
                << (dtrees[0].sampling.statisticalWeight() + dtrees[1].sampling.statisticalWeight()) / 2 << ' '
                << (dtrees[2].sampling.statisticalWeight() + dtrees[3].sampling.statisticalWeight()) / 2 << std::endl;
        }

        ofs << this->dtree_export_results.c_str();
        ofs << "-1" << std::endl;

        mixNodes.clear();
        mixNodes.emplace_back();
        for (int i = 0; i < N; ++i) m_atomic[i].statisticalWeight = m_atomic[i].sum = 0;
    }

    uint64_t size() const
    {
        uint64_t res = 0;
        for (int i = 0; i < N; ++i) res += dtrees[i].approxMemoryFootprint();
        res += sizeof(m_atomic);
        res += sizeof(isLeaf);
        res += sizeof(axis);
        res += 2 * sizeof(uint32_t);  // children
        return res;
    }
};

using STreeNode3PRB = STreeNodeN<3, true>;
using STreeNode5 = STreeNodeN<5, false>;


template<typename>
struct getDTreeNumPerSTreeNode_impl : std::integral_constant<int, 0> {};
template<int N, bool prb>
struct getDTreeNumPerSTreeNode_impl<STreeNodeN<N, prb>> : std::integral_constant<int, N> {};
template<typename T>
constexpr int getDTreeNumPerSTreeNode_v = getDTreeNumPerSTreeNode_impl<T>::value;

template<typename>
struct isSTreeNodeForPRB_impl : std::false_type {};
template<int N, bool prb>
struct isSTreeNodeForPRB_impl<STreeNodeN<N, prb>> : std::conditional_t<prb, std::true_type, std::false_type> {};
template<typename T>
constexpr bool isSTreeNodeForPRB_v = isSTreeNodeForPRB_impl<T>::value;

template<typename>
struct isSpecializedStreeNodeN_impl : std::false_type {};
template<int N, bool prb>
struct isSpecializedStreeNodeN_impl<STreeNodeN<N, prb>> : std::true_type {};
template<typename T>
constexpr bool isSpecializedStreeNodeN_v = isSpecializedStreeNodeN_impl<T>::value;   // for us, here do not need sth like std::decay_t<T>


template<typename SNode>
class STree {
    static_assert(std::is_same_v<SNode, STreeNode> || isSpecializedStreeNodeN_v<SNode>);
public:
    STree(const AABB& aabb) {

        clear();

        m_aabb = aabb;

        // // Enlarge AABB to turn it into a cube. This has the effect
        // // of nicer hierarchical subdivisions.
        // if (g_stepper == 0) {
        //     Vector size = m_aabb.max - m_aabb.min;
        //     Float maxSize = std::max(std::max(size.x, size.y), size.z);
        //     m_aabb.max = m_aabb.min + Vector(maxSize);
        // }
        // else {
        //     Vector size = m_aabb.max - m_aabb.min;
        //     Vector delta = size * 0.0015;
        //     m_aabb.min += delta;
        //     m_aabb.max -= delta;
        // }
    }

    void clear() {
        m_nodes.clear();
        m_nodes.emplace_back();
    }

    void subdivideAll() {
        int nNodes = (int)m_nodes.size();
        for (int i = 0; i < nNodes; ++i) {
            if (m_nodes[i].isLeaf) {
                subdivide(i, m_nodes);
            }
        }
    }

    void subdivide(int nodeIdx, std::vector<SNode>& nodes) {
        // Add 2 child nodes
        nodes.resize(nodes.size() + 2);

        if (nodes.size() > std::numeric_limits<uint32_t>::max()) {
            SLog(EWarn, "DTreeWrapper hit maximum children count.");
            return;
        }

        if constexpr (std::is_same_v<SNode, STreeNode>) {
            STreeNode& cur = nodes[nodeIdx];
            for (int i = 0; i < 2; ++i) {
                uint32_t idx = (uint32_t)nodes.size() - 2 + i;
                cur.children[i] = idx;
                nodes[idx].axis = (cur.axis + 1) % 3;
                nodes[idx].dTree = cur.dTree;
                nodes[idx].dTree.setStatisticalWeightBuilding(nodes[idx].dTree.statisticalWeightBuilding() / 2);
                nodes[idx].dTree.setStatisticalWeightSampling(nodes[idx].dTree.statisticalWeight() / 2);
                nodes[idx].dTree.scale(0.5);
            }
            cur.isLeaf = false;
            cur.dTree = {}; // Reset to an empty dtree to save memory.
        }
        else {
            auto& cur = nodes[nodeIdx];
            for (int i = 0; i < 2; ++i) {
                uint32_t idx = (uint32_t)nodes.size() - 2 + i;
                cur.children[i] = idx;
                nodes[idx].axis = (cur.axis + 1) % 3;
                for(int j = 0; j < getDTreeNumPerSTreeNode_v<SNode>; ++j) {
                    nodes[idx].dtrees[j] = cur.dtrees[j];
                    nodes[idx].dtrees[j].setStatisticalWeightBuilding(nodes[idx].dtrees[j].statisticalWeightBuilding() / 2);
                    nodes[idx].dtrees[j].setStatisticalWeightSampling(nodes[idx].dtrees[j].statisticalWeight() / 2);
                    nodes[idx].dtrees[j].scale(0.5);
                }
            }
            cur.isLeaf = false;
            for (int i = 0; i < getDTreeNumPerSTreeNode_v<SNode>; ++i) cur.dtrees[i] = {};
        }
    }

    DTreeWrapper* dTreeWrapper(Point p, Vector& size, EDTreeType dtree_type = EDTreeType::EPrimal) {
        size = m_aabb.getExtents();
        p = Point(p - m_aabb.min);
        p.x /= size.x;
        p.y /= size.y;
        p.z /= size.z;

        return m_nodes[0].dTreeWrapper(p, size, m_nodes, dtree_type);
    }

    DTreeWrapper* dTreeWrapper(Point p, EDTreeType dtree_type = EDTreeType::EPrimal) {
        Vector size;
        return dTreeWrapper(p, size, dtree_type);
    }

    void forEachDTreeWrapperConst(std::function<void(const DTreeWrapper*)> func) const {
        for (auto& node : m_nodes) {
            if (node.isLeaf) {
                if constexpr (std::is_same_v<SNode, STreeNode>) {
                    func(&node.dTree);
                }
                else {
                    for (const auto& dtree : node.dtrees)
                        func(&dtree);
                }
            }
        }
    }

    void forEachDTreeWrapperConstP(std::function<void(const DTreeWrapper*, const Point&, const Vector&)> func) const {
        m_nodes[0].forEachLeaf(func, m_aabb.min, m_aabb.max - m_aabb.min, m_nodes);
    }

    void forEachDTreeWrapperParallel(std::function<void(DTreeWrapper*)> func) {
        int nDTreeWrappers = static_cast<int>(m_nodes.size());

#pragma omp parallel for
        for (int i = 0; i < nDTreeWrappers; ++i) {
            if (m_nodes[i].isLeaf) {
                if constexpr (std::is_same_v<SNode, STreeNode>) {
                    func(&m_nodes[i].dTree);
                }
                else {
                    for (auto& dtree : m_nodes[i].dtrees)
                        func(&dtree);
                }
            }
        }
    }

    void record(const Point& p, const Vector& dTreeVoxelSize, DTreeRecord rec,
        EDirectionalFilter directionalFilter, EBsdfSamplingFractionLoss bsdfSamplingFractionLoss,
        EDTreeType dtree_type = EDTreeType::EPrimal) {
        Float volume = 1;
        for (int i = 0; i < 3; ++i) {
            volume *= dTreeVoxelSize[i];
        }

        rec.statisticalWeight /= volume;
        m_nodes[0].record(p - dTreeVoxelSize * 0.5f, p + dTreeVoxelSize * 0.5f, m_aabb.min, m_aabb.getExtents(), rec, directionalFilter, bsdfSamplingFractionLoss, m_nodes, dtree_type);
    }

    void dump(BlobWriter& blob) const {
        forEachDTreeWrapperConstP([&blob](const DTreeWrapper* dTree, const Point& p, const Vector& size) {
            if (dTree->statisticalWeight() > 0) {
                dTree->dump(blob, p, size);
            }
            });
    }

    bool shallSplit(const SNode& node, int depth, size_t samplesRequired) {
        if constexpr (std::is_same_v<SNode, STreeNode>) {
            return m_nodes.size() < std::numeric_limits<uint32_t>::max() - 1 && node.dTree.statisticalWeightBuilding() > samplesRequired;
        }
        else {
            Float swb = 0;
            if constexpr (isSTreeNodeForPRB_v<SNode>) {
                swb = node.dtrees[SNode::dtreeTypeToIdx(EDTreeType::EPrimal)].statisticalWeightBuilding();
            }
            else {
            // TODO use what for SRB ?
                for (const auto& dtree : node.dtrees) swb += dtree.statisticalWeightBuilding();  // all samples
                //swb = node.dtrees[SNode::dtreeTypeToIdx(EDTreeType::ELf)].statisticalWeightBuilding();
            }
            return m_nodes.size() < std::numeric_limits<uint32_t>::max() - 1 && swb > samplesRequired;
        }
    }

    void refine(size_t sTreeThreshold, int maxMB) {
        //if (m_nodes.size() > 1) {
        //    return;
        //}
        if (maxMB >= 0) {
            size_t approxMemoryFootprint = 0;
            for (const auto& node : m_nodes) {
                if constexpr (std::is_same_v<SNode, STreeNode>) {
                    approxMemoryFootprint += node.dTreeWrapper()->approxMemoryFootprint();
                }
                else {
                    for (const auto& dtree : node.dtrees) approxMemoryFootprint += dtree.approxMemoryFootprint();
                }
            }

            if (approxMemoryFootprint / 1000000 >= (size_t)maxMB) {
                return;
            }
        }

        struct StackNode {
            size_t index;
            int depth;
        };

        std::stack<StackNode> nodeIndices;
        nodeIndices.push({ 0,  1 });
        while (!nodeIndices.empty()) {
            StackNode sNode = nodeIndices.top();
            nodeIndices.pop();

            // Subdivide if needed and leaf
            if (m_nodes[sNode.index].isLeaf) {
                if (shallSplit(m_nodes[sNode.index], sNode.depth, sTreeThreshold)) {
                    subdivide((int)sNode.index, m_nodes);
                }
            }

            // Add children to stack if we're not
            if (!m_nodes[sNode.index].isLeaf) {
                const auto& node = m_nodes[sNode.index];
                for (int i = 0; i < 2; ++i) {
                    nodeIndices.push({ node.children[i], sNode.depth + 1 });
                }
            }
        }

        // Uncomment once memory becomes an issue.
        //m_nodes.shrink_to_fit();
        std::cout << "    STree size = " << m_nodes.size() << std::endl;
        set_anypos();
    }

    const AABB& aabb() const {
        return m_aabb;
    }

    void my_vis_dfs(int i, Point3f pos, Vector3f siz) const {
        std::stringstream ss;
        ss << "Node " << i << " ";
        ss << "axis=" << m_nodes[i].axis << " ";
        ss << "ch0=" << m_nodes[i].children[0] << " ";
        ss << "ch1=" << m_nodes[i].children[1] << " ";
        ss << "pos=" << pos.toString() << " ";
        ss << "size=" << siz.toString() << " ";

        Float swb = 0;
        if constexpr (std::is_same_v<SNode, STreeNode>) {
            swb = m_nodes[i].dTreeWrapper()->statisticalWeightBuilding();
        }
        else {
            for (const auto& dtree : m_nodes[i].dtrees) swb += dtree.statisticalWeightBuilding();
        }
        ss << "statWeight=" << swb << " ";
        std::cout << ss.str() << std::endl;

        if constexpr (std::is_same_v<SNode, STreeNode>) {
            m_nodes[i].dTreeWrapper()->my_vis();
        }
        else {
            for (const auto& dtree : m_nodes[i].dtrees) dtree.my_vis();
        }

        if (m_nodes[i].isLeaf) return;

        siz[m_nodes[i].axis] /= 2;
        my_vis_dfs(m_nodes[i].children[0], pos, siz);
        pos[m_nodes[i].axis] += siz[m_nodes[i].axis];
        my_vis_dfs(m_nodes[i].children[1], pos, siz);
    }

    void my_vis() const {
        std::cout << "Vis STree size=" << m_nodes.size() << std::endl;
        my_vis_dfs(0, m_aabb.getCorner(0), m_aabb.getExtents());
    }

    void set_anypos_dfs(int i, Point3f pos, Vector3f siz) {
        if constexpr (std::is_same_v<SNode, STreeNode>) {
            m_nodes[i].dTreeWrapperNonconst()->any_position = Point3f(pos + siz * 0.5);
        }
        else {
            for (auto& dtree : m_nodes[i].dtrees)
                dtree.any_position = Point3f(pos + siz * 0.5);
        }

        if (m_nodes[i].isLeaf) return;

        siz[m_nodes[i].axis] /= 2;
        set_anypos_dfs(m_nodes[i].children[0], pos, siz);
        pos[m_nodes[i].axis] += siz[m_nodes[i].axis];
        set_anypos_dfs(m_nodes[i].children[1], pos, siz);
    }

    void set_anypos() {
        set_anypos_dfs(0, m_aabb.getCorner(0), m_aabb.getExtents());
    }

    void my_export_dfs(int i, Point3f pos, Vector3f siz, OFSTREAM& ofs) const {
       
        
        ofs << i << " ";
        ofs << m_nodes[i].axis << " ";
        ofs << (int)m_nodes[i].children[0] << " ";
        ofs << (int)m_nodes[i].children[1] << " ";
        ofs << pos[0] << " ";
        ofs << pos[1] << " ";
        ofs << pos[2] << " ";
        ofs << siz[0] << " ";
        ofs << siz[1] << " ";
        ofs << siz[2] << " ";
        ofs << std::endl;
        if constexpr (std::is_same_v<SNode, STreeNode>) {
            m_nodes[i].dTreeWrapper()->my_export(ofs);
        }
        else {
            m_nodes[i].my_export(ofs);
        }


        if (m_nodes[i].isLeaf) return;

        siz[m_nodes[i].axis] /= 2;
        my_export_dfs(m_nodes[i].children[0], pos, siz, ofs);
        pos[m_nodes[i].axis] += siz[m_nodes[i].axis];
        my_export_dfs(m_nodes[i].children[1], pos, siz, ofs);
    }

    void prepare_before_export() const {
        if constexpr (isSpecializedStreeNodeN_v<SNode>) {
        
#pragma omp parallel for
            for (int i = 0; i < m_nodes.size(); ++i) {
                if (m_nodes[i].isLeaf) {
                    m_nodes[i].generate_merged_result();
                }
            }

        }
    }

    void parallel_export_dtree(OFSTREAM & ofs) {
        if constexpr (isSpecializedStreeNodeN_v<SNode>) {

#pragma omp parallel for
            for (int i = 0; i < m_nodes.size(); ++i) {
                m_nodes[i].my_export_dtree(ofs);
            }

        }
    }
    
    void my_export(OFSTREAM& ofs)  {
        //std::clock_t start_time = std::clock();
        prepare_before_export();
        //std::cout << "prepare time uses: " << std::clock() - start_time << std::endl;
        //start_time = std::clock();
        parallel_export_dtree(ofs);
        //std::cout << "parallel export dtree uses: " << std::clock() - start_time << std::endl;
        //start_time = std::clock();
        ofs << m_nodes.size() << std::endl;
        my_export_dfs(0, m_aabb.getCorner(0), m_aabb.getExtents(), ofs);
        //std::cout << "traverse stree and export dfs time uses: " << std::clock() - start_time << std::endl;
        //start_time = std::clock();
        ofs << -1 << std::endl;
        my_export_lut(ofs);
       // std::cout << "export lut time uses: " << std::clock() - start_time << std::endl;

        // output sdtree size
        if constexpr (isSpecializedStreeNodeN_v<SNode>) {
            uint64_t sz = 0;
            for (int i = 0; i < m_nodes.size(); ++i) {
                sz += m_nodes[i].size();
            }
            std::cout << "Total SDTree Size: " << sz << " in bytes\n";
        }
    }

    void my_export_lut(OFSTREAM &ofs) const
    {
        // todo: parallel
        int resl = 5;
        int res = 1 << resl;
        for (int i = 0; i < res; i++)
        {
            for (int j = 0; j < res; j++)
            {
                for (int k = 0; k < res; k++)
                {
                    double x = (i + 0.5) / res;
                    double y = (j + 0.5) / res;
                    double z = (k + 0.5) / res;
                    Point3f pos = Point3f(z, y, x);
                    int p = m_nodes[0].getNodeId(pos, m_nodes, 0, resl * 3);
                    ofs << p << " ";
                }
                ofs << std::endl;
            }
        }
    }

    void my_import(std::ifstream& ifs) {
        int n;
        ifs >> n;

        // Do not clear, because we need to trace the building and fused distributions
        // The refine process is always happened inside mts1
        // m_nodes.clear();
        // m_nodes.resize(n);

        int p, ax, p0, p1;
        while (true) {
            ifs >> p;
            if (p == -1) {
                break;
            }

            ifs >> ax >> p0 >> p1;

            m_nodes[p].axis = ax;
            m_nodes[p].children[0] = p0;
            m_nodes[p].children[1] = p1;


            if constexpr (std::is_same_v<SNode, STreeNode>) {
                m_nodes[p].dTreeWrapperNonconst()->my_import(ifs);
            }
            else {
                // TODO not sure what to do for STreeNodeN
            }
        }
    }


private:
    std::vector<SNode> m_nodes;
    AABB m_aabb;
};

static size_t getFileSize(FILE* file)
{
    fseek(file, 0, SEEK_END);
    size_t read_len = ftell(file);
    fseek(file, 0, SEEK_SET);
    return read_len;
}

std::vector<unsigned char> readFromFile(const char* filePath)
{
    std::vector<unsigned char> result;
    FILE* file = fopen(filePath, "rb");
    if (file == nullptr) {
        return result;
    }

    size_t fileSize = getFileSize(file);
    if (fileSize != 0) {
        result.resize(fileSize);
        // read all at once
        fread(result.data(), 1, fileSize, file);
        // assume no error   
    }
    fclose(file);

    return result;
}


static StatsCounter avgPathLength("Guided path tracer", "Average path length", EAverage);

class GuidedPathTracer : public MonteCarloIntegrator {
public:
    GuidedPathTracer(const Properties &props) : MonteCarloIntegrator(props) {
        m_neeStr = props.getString("nee", "never");
        if (m_neeStr == "never") {
            m_nee = ENever;
        } else if (m_neeStr == "kickstart") {
            m_nee = EKickstart;
        } else if (m_neeStr == "always") {
            m_nee = EAlways;
        } else {
            Assert(false);
        }

        m_sampleCombinationStr = props.getString("sampleCombination", "automatic");
        if (m_sampleCombinationStr == "discard") {
            m_sampleCombination = ESampleCombination::EDiscard;
        } else if (m_sampleCombinationStr == "automatic") {
            m_sampleCombination = ESampleCombination::EDiscardWithAutomaticBudget;
        } else if (m_sampleCombinationStr == "inversevar") {
            m_sampleCombination = ESampleCombination::EInverseVariance;
        } else {
            m_sampleCombination = ESampleCombination::EDiscard;
        }

        m_sampleAllocSeqStr = props.getString("sampleAllocSeq", "double");
        if (m_sampleAllocSeqStr == "double") {
            m_sampleAllocSeq = ESampleAllocSeq::EDouble;
        } else if (m_sampleAllocSeqStr == "halfdouble") {
            m_sampleAllocSeq = ESampleAllocSeq::EHalfdouble;
        } else if (m_sampleAllocSeqStr == "uniform") {
            m_sampleAllocSeq = ESampleAllocSeq::EUniform;
        } else {
            Assert(false);
        }


        m_spatialFilterStr = props.getString("spatialFilter_Primal", "nearest");
        if (m_spatialFilterStr == "nearest") {
            m_spatialFilter_Primal = ESpatialFilter::ENearest;
        }
        else if (m_spatialFilterStr == "stochastic") {
            m_spatialFilter_Primal = ESpatialFilter::EStochasticBox;
        }
        else if (m_spatialFilterStr == "box") {
            m_spatialFilter_Primal = ESpatialFilter::EBox;
        }
        else {
            Assert(false);
        }
        m_spatialFilterStr = props.getString("spatialFilter_Adjoint", "nearest");
        if (m_spatialFilterStr == "nearest") {
            m_spatialFilter_Adjoint = ESpatialFilter::ENearest;
        }
        else if (m_spatialFilterStr == "stochastic") {
            m_spatialFilter_Adjoint = ESpatialFilter::EStochasticBox;
        }
        else if (m_spatialFilterStr == "box") {
            m_spatialFilter_Adjoint = ESpatialFilter::EBox;
        }
        else {
            Assert(false);
        }
        m_spatialFilter = m_spatialFilter_Adjoint;
        //Log(ELogLevel::EInfo, "sfilter %d %d\n", int(m_spatialFilter_Primal), int(m_spatialFilter_Adjoint));

        m_directionalFilterStr = props.getString("directionalFilter", "nearest");
        if (m_directionalFilterStr == "nearest") {
            m_directionalFilter = EDirectionalFilter::ENearest;
        } else if (m_directionalFilterStr == "box") {
            m_directionalFilter = EDirectionalFilter::EBox;
        } else {
            Assert(false);
        }

        m_bsdfSamplingFractionLossStr = props.getString("bsdfSamplingFractionLoss", "none");
        if (m_bsdfSamplingFractionLossStr == "none") {
            m_bsdfSamplingFractionLoss = EBsdfSamplingFractionLoss::ENone;
        } else if (m_bsdfSamplingFractionLossStr == "kl") {
            m_bsdfSamplingFractionLoss = EBsdfSamplingFractionLoss::EKL;
        } else if (m_bsdfSamplingFractionLossStr == "var") {
            m_bsdfSamplingFractionLoss = EBsdfSamplingFractionLoss::EVariance;
        } else {
            Assert(false);
        }

        m_sdTreeMaxMemory = props.getInteger("sdTreeMaxMemory", -1);
        m_sTreeThreshold = props.getInteger("sTreeThreshold", 4000);
        m_dTreeThreshold = props.getFloat("dTreeThreshold", 0.01f);
        m_bsdfSamplingFraction = props.getFloat("bsdfSamplingFraction", 0.5f);
        m_sppPerPass = props.getInteger("sppPerPass", 4);

        m_budgetStr = props.getString("budgetType", "seconds");
        if (m_budgetStr == "spp") {
            m_budgetType = ESpp;
        } else if (m_budgetStr == "seconds") {
            m_budgetType = ESeconds;
        } else {
            Assert(false);
        }

        m_budget = props.getFloat("budget", 300.0f);
        m_dumpSDTree = props.getBoolean("dumpSDTree", false);

        m_lazyRebuild = props.getBoolean("lazyRebuild", false);
        m_freqRebuild = props.getBoolean("freqRebuild", false);
        m_filteredGlints = props.getBoolean("filteredGlints", false);
        m_trainingFusion = props.getBoolean("trainingFusion", false);
        m_workingMode = props.getInteger("stepper", 0);
        g_stepper = m_workingMode;
        m_adjointTarget = props.getInteger("adjointTarget", 0);
        //m_decomposeAdjoint = m_adjointTarget >= 3;
        m_useMixSDTree = m_adjointTarget == 4;
        m_useSDTree_SRB5 = m_adjointTarget == 5;
        if (m_workingMode == 1) {
            Assert(m_useMixSDTree || m_useSDTree_SRB5);
        }

        m_tempParam = props.getFloat("tempParam", 0);
        global_temp_param = m_tempParam;

        m_spatialFilteringRadius = props.getFloat("spatialFilteringRadius", 20);
        m_spatialFilteringRepeat = props.getInteger("spatialFilteringRepeat", 12);
        m_temporalFilteringWeight = props.getFloat("temporalFilteringWeight", 0.3);
        global_guiding_mis_weight_ad = props.getFloat("guiding_mis_weight_ad", 0.5);
        global_decay_rate = props.getFloat("decay_rate", 1.0);

        float bbox_cx = props.getFloat("bbox_cx", 0);
        float bbox_cy = props.getFloat("bbox_cy", 0);
        float bbox_cz = props.getFloat("bbox_cz", 0);
        float bbox_sx = props.getFloat("bbox_sx", 0);
        float bbox_sy = props.getFloat("bbox_sy", 0);
        float bbox_sz = props.getFloat("bbox_sz", 0);
        m_aabb = AABB(Point3f(bbox_cx - bbox_sx, bbox_cy - bbox_sy, bbox_cz - bbox_sz), 
                      Point3f(bbox_cx + bbox_sx, bbox_cy + bbox_sy, bbox_cz + bbox_sz));
    }

    ref<BlockedRenderProcess> renderPass(Scene *scene,
        RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID, int integratorResID) {

       
        ref<BlockedRenderProcess> proc = new BlockedRenderProcess(job,
            queue, scene->getBlockSize());

        proc->disableProgress();

        proc->bindResource("integrator", integratorResID);
        proc->bindResource("scene", sceneResID);
        proc->bindResource("sensor", sensorResID);
        proc->bindResource("sampler", samplerResID);

        scene->bindUsedResources(proc);
        bindUsedResources(proc);

        return proc;
    }

    void initSDTree() {
        if (m_useMixSDTree) {
            m_sdTreeMix = std::make_shared<STree<STreeNode3PRB>>(m_aabb);
        }
        if (m_useSDTree_SRB5) {
            m_sdTreeSRB5 = std::make_shared<STree<STreeNode5>>(m_aabb);
        }
    }

    void resetSDTree() {
        HDTimer timer;
        // Log(EInfo, "Resetting distributions for sampling.");

        if (m_workingMode == 0) {
            m_sdTree->refine((size_t)(std::sqrt(std::pow(2, m_iter) * m_sppPerPass / 4) * m_sTreeThreshold), m_sdTreeMaxMemory);
        }
        else {
            // emperically using a smaller thres for adjoint distribution
           
            if (m_useMixSDTree) {
                m_sdTreeMix->refine((size_t)(m_sTreeThreshold), m_sdTreeMaxMemory);
            }
            if (m_useSDTree_SRB5) {
                m_sdTreeSRB5->refine((size_t)(m_sTreeThreshold), m_sdTreeMaxMemory);
            }
        }
        if (m_workingMode == 0) {
            m_sdTree->forEachDTreeWrapperParallel([this](DTreeWrapper* dTree) { dTree->reset(10, m_dTreeThreshold); });
        }
        if (m_workingMode == 1) {
           
            if (m_useMixSDTree) {
                m_sdTreeMix->forEachDTreeWrapperParallel([this](DTreeWrapper* dTree) { dTree->reset(10, m_dTreeThreshold); });
            }
            if (m_useSDTree_SRB5) {
                m_sdTreeSRB5->forEachDTreeWrapperParallel([this](DTreeWrapper* dTree) { dTree->reset(10, m_dTreeThreshold); });
            }
        }
        Log(EInfo, "reset tree time %f s", timer.value());
    }

    void buildSDTree() {
        HDTimer timer;

        // Log(EInfo, "Building distributions for sampling.");

        // Build distributions
        if (m_workingMode == 0) {
            m_sdTree->forEachDTreeWrapperParallel([](DTreeWrapper* dTree) { dTree->build(); });
        }
        if (m_workingMode == 1) {
           
            if (m_useMixSDTree) {
                m_sdTreeMix->forEachDTreeWrapperParallel([](DTreeWrapper* dTree) { dTree->build(); });
            }
            if (m_useSDTree_SRB5) {
                m_sdTreeSRB5->forEachDTreeWrapperParallel([](DTreeWrapper* dTree) { dTree->build(); });
            }
        }
        Log(EInfo, "build tree time %f s", timer.value());


        // // Gather statistics
        // int maxDepth = 0;
        // int minDepth = std::numeric_limits<int>::max();
        // Float avgDepth = 0;
        // Float maxAvgRadiance = 0;
        // Float minAvgRadiance = std::numeric_limits<Float>::max();
        // Float avgAvgRadiance = 0;
        // size_t maxNodes = 0;
        // size_t minNodes = std::numeric_limits<size_t>::max();
        // Float avgNodes = 0;
        // Float maxStatisticalWeight = 0;
        // Float minStatisticalWeight = std::numeric_limits<Float>::max();
        // Float avgStatisticalWeight = 0;

        // int nPoints = 0;
        // int nPointsNodes = 0;

        // m_sdTree->forEachDTreeWrapperConst([&](const DTreeWrapper* dTree) {
        //     const int depth = dTree->depth();
        //     maxDepth = std::max(maxDepth, depth);
        //     minDepth = std::min(minDepth, depth);
        //     avgDepth += depth;

        //     const Float avgRadiance = dTree->meanRadiance();
        //     maxAvgRadiance = std::max(maxAvgRadiance, avgRadiance);
        //     minAvgRadiance = std::min(minAvgRadiance, avgRadiance);
        //     avgAvgRadiance += avgRadiance;

        //     if (dTree->numNodes() > 1) {
        //         const size_t nodes = dTree->numNodes();
        //         maxNodes = std::max(maxNodes, nodes);
        //         minNodes = std::min(minNodes, nodes);
        //         avgNodes += nodes;
        //         ++nPointsNodes;
        //     }

        //     const Float statisticalWeight = dTree->statisticalWeight();
        //     maxStatisticalWeight = std::max(maxStatisticalWeight, statisticalWeight);
        //     minStatisticalWeight = std::min(minStatisticalWeight, statisticalWeight);
        //     avgStatisticalWeight += statisticalWeight;

        //     ++nPoints;
        // });

        // if (nPoints > 0) {
        //     avgDepth /= nPoints;
        //     avgAvgRadiance /= nPoints;

        //     if (nPointsNodes > 0) {
        //         avgNodes /= nPointsNodes;
        //     }

        //     avgStatisticalWeight /= nPoints;
        // }

        // Log(EInfo,
        //     "Distribution statistics:\n"
        //     "  Depth         = [%d, %f, %d]\n"
        //     "  Mean radiance = [%f, %f, %f]\n"
        //     "  Node count    = [" SIZE_T_FMT ", %f, " SIZE_T_FMT "]\n"
        //     "  Stat. weight  = [%f, %f, %f]\n",
        //     minDepth, avgDepth, maxDepth,
        //     minAvgRadiance, avgAvgRadiance, maxAvgRadiance,
        //     minNodes, avgNodes, maxNodes,
        //     minStatisticalWeight, avgStatisticalWeight, maxStatisticalWeight
        // );


        // if (m_workingMode == 1) {
        //     // Gather statistics
        //     int maxDepth = 0;
        //     int minDepth = std::numeric_limits<int>::max();
        //     Float avgDepth = 0;
        //     Float maxAvgRadiance = 0;
        //     Float minAvgRadiance = std::numeric_limits<Float>::max();
        //     Float avgAvgRadiance = 0;
        //     size_t maxNodes = 0;
        //     size_t minNodes = std::numeric_limits<size_t>::max();
        //     Float avgNodes = 0;
        //     Float maxStatisticalWeight = 0;
        //     Float minStatisticalWeight = std::numeric_limits<Float>::max();
        //     Float avgStatisticalWeight = 0;

        //     int nPoints = 0;
        //     int nPointsNodes = 0;

        //     m_sdTreePrimalIncremental->forEachDTreeWrapperConst([&](const DTreeWrapper* dTree) {
        //         const int depth = dTree->depth();
        //         maxDepth = std::max(maxDepth, depth);
        //         minDepth = std::min(minDepth, depth);
        //         avgDepth += depth;

        //         const Float avgRadiance = dTree->meanRadiance();
        //         maxAvgRadiance = std::max(maxAvgRadiance, avgRadiance);
        //         minAvgRadiance = std::min(minAvgRadiance, avgRadiance);
        //         avgAvgRadiance += avgRadiance;

        //         if (dTree->numNodes() > 1) {
        //             const size_t nodes = dTree->numNodes();
        //             maxNodes = std::max(maxNodes, nodes);
        //             minNodes = std::min(minNodes, nodes);
        //             avgNodes += nodes;
        //             ++nPointsNodes;
        //         }

        //         const Float statisticalWeight = dTree->statisticalWeight();
        //         maxStatisticalWeight = std::max(maxStatisticalWeight, statisticalWeight);
        //         minStatisticalWeight = std::min(minStatisticalWeight, statisticalWeight);
        //         avgStatisticalWeight += statisticalWeight;

        //         ++nPoints;
        //     });

        //     if (nPoints > 0) {
        //         avgDepth /= nPoints;
        //         avgAvgRadiance /= nPoints;

        //         if (nPointsNodes > 0) {
        //             avgNodes /= nPointsNodes;
        //         }

        //         avgStatisticalWeight /= nPoints;
        //     }

        //     Log(EInfo,
        //         "SDThis Distribution statistics:\n"
        //         "  Depth         = [%d, %f, %d]\n"
        //         "  Mean radiance = [%f, %f, %f]\n"
        //         "  Node count    = [" SIZE_T_FMT ", %f, " SIZE_T_FMT "]\n"
        //         "  Stat. weight  = [%f, %f, %f]\n",
        //         minDepth, avgDepth, maxDepth,
        //         minAvgRadiance, avgAvgRadiance, maxAvgRadiance,
        //         minNodes, avgNodes, maxNodes,
        //         minStatisticalWeight, avgStatisticalWeight, maxStatisticalWeight
        //     );
        // }


        // if (m_workingMode == 1) {
        //     // Gather statistics
        //     int maxDepth = 0;
        //     int minDepth = std::numeric_limits<int>::max();
        //     Float avgDepth = 0;
        //     Float maxAvgRadiance = 0;
        //     Float minAvgRadiance = std::numeric_limits<Float>::max();
        //     Float avgAvgRadiance = 0;
        //     size_t maxNodes = 0;
        //     size_t minNodes = std::numeric_limits<size_t>::max();
        //     Float avgNodes = 0;
        //     Float maxStatisticalWeight = 0;
        //     Float minStatisticalWeight = std::numeric_limits<Float>::max();
        //     Float avgStatisticalWeight = 0;

        //     int nPoints = 0;
        //     int nPointsNodes = 0;

        //     m_sdTreeAdjointIncremental->forEachDTreeWrapperConst([&](const DTreeWrapper* dTree) {
        //         const int depth = dTree->depth();
        //         maxDepth = std::max(maxDepth, depth);
        //         minDepth = std::min(minDepth, depth);
        //         avgDepth += depth;

        //         const Float avgRadiance = dTree->meanRadiance();
        //         maxAvgRadiance = std::max(maxAvgRadiance, avgRadiance);
        //         minAvgRadiance = std::min(minAvgRadiance, avgRadiance);
        //         avgAvgRadiance += avgRadiance;

        //         if (dTree->numNodes() > 1) {
        //             const size_t nodes = dTree->numNodes();
        //             maxNodes = std::max(maxNodes, nodes);
        //             minNodes = std::min(minNodes, nodes);
        //             avgNodes += nodes;
        //             ++nPointsNodes;
        //         }

        //         const Float statisticalWeight = dTree->statisticalWeight();
        //         maxStatisticalWeight = std::max(maxStatisticalWeight, statisticalWeight);
        //         minStatisticalWeight = std::min(minStatisticalWeight, statisticalWeight);
        //         avgStatisticalWeight += statisticalWeight;

        //         ++nPoints;
        //     });

        //     if (nPoints > 0) {
        //         avgDepth /= nPoints;
        //         avgAvgRadiance /= nPoints;

        //         if (nPointsNodes > 0) {
        //             avgNodes /= nPointsNodes;
        //         }

        //         avgStatisticalWeight /= nPoints;
        //     }

        //     Log(EInfo,
        //         "SDT2his Distribution statistics:\n"
        //         "  Depth         = [%d, %f, %d]\n"
        //         "  Mean radiance = [%f, %f, %f]\n"
        //         "  Node count    = [" SIZE_T_FMT ", %f, " SIZE_T_FMT "]\n"
        //         "  Stat. weight  = [%f, %f, %f]\n",
        //         minDepth, avgDepth, maxDepth,
        //         minAvgRadiance, avgAvgRadiance, maxAvgRadiance,
        //         minNodes, avgNodes, maxNodes,
        //         minStatisticalWeight, avgStatisticalWeight, maxStatisticalWeight
        //     );
        // }


        // if (m_workingMode == 1) {
        //     // Gather statistics
        //     int maxDepth = 0;
        //     int minDepth = std::numeric_limits<int>::max();
        //     Float avgDepth = 0;
        //     Float maxAvgRadiance = 0;
        //     Float minAvgRadiance = std::numeric_limits<Float>::max();
        //     Float avgAvgRadiance = 0;
        //     size_t maxNodes = 0;
        //     size_t minNodes = std::numeric_limits<size_t>::max();
        //     Float avgNodes = 0;
        //     Float maxStatisticalWeight = 0;
        //     Float minStatisticalWeight = std::numeric_limits<Float>::max();
        //     Float avgStatisticalWeight = 0;

        //     int nPoints = 0;
        //     int nPointsNodes = 0;

        //     m_sdTree2->forEachDTreeWrapperConst([&](const DTreeWrapper* dTree) {
        //         const int depth = dTree->depth();
        //         maxDepth = std::max(maxDepth, depth);
        //         minDepth = std::min(minDepth, depth);
        //         avgDepth += depth;

        //         const Float avgRadiance = dTree->meanRadiance();
        //         maxAvgRadiance = std::max(maxAvgRadiance, avgRadiance);
        //         minAvgRadiance = std::min(minAvgRadiance, avgRadiance);
        //         avgAvgRadiance += avgRadiance;

        //         if (dTree->numNodes() > 1) {
        //             const size_t nodes = dTree->numNodes();
        //             maxNodes = std::max(maxNodes, nodes);
        //             minNodes = std::min(minNodes, nodes);
        //             avgNodes += nodes;
        //             ++nPointsNodes;
        //         }

        //         const Float statisticalWeight = dTree->statisticalWeight();
        //         maxStatisticalWeight = std::max(maxStatisticalWeight, statisticalWeight);
        //         minStatisticalWeight = std::min(minStatisticalWeight, statisticalWeight);
        //         avgStatisticalWeight += statisticalWeight;

        //         ++nPoints;
        //     });

        //     if (nPoints > 0) {
        //         avgDepth /= nPoints;
        //         avgAvgRadiance /= nPoints;

        //         if (nPointsNodes > 0) {
        //             avgNodes /= nPointsNodes;
        //         }

        //         avgStatisticalWeight /= nPoints;
        //     }

        //     Log(EInfo,
        //         "Distribution statistics:\n"
        //         "  Depth         = [%d, %f, %d]\n"
        //         "  Mean radiance = [%f, %f, %f]\n"
        //         "  Node count    = [" SIZE_T_FMT ", %f, " SIZE_T_FMT "]\n"
        //         "  Stat. weight  = [%f, %f, %f]\n",
        //         minDepth, avgDepth, maxDepth,
        //         minAvgRadiance, avgAvgRadiance, maxAvgRadiance,
        //         minNodes, avgNodes, maxNodes,
        //         minStatisticalWeight, avgStatisticalWeight, maxStatisticalWeight
        //     );
        // }

        m_isBuilt = true;
    }


    void buildpostSDTree() {
        // Build distributions
        // m_sdTree->forEachDTreeWrapperParallel([](DTreeWrapper* dTree) { dTree->buildpost(); });
        // if (m_workingMode == 1) {
        //     m_sdTree2->forEachDTreeWrapperParallel([](DTreeWrapper* dTree) { dTree->buildpost(); });
        //     m_sdTreePrimalIncremental->forEachDTreeWrapperParallel([](DTreeWrapper* dTree) { dTree->buildpost(); });
        //     m_sdTreeAdjointIncremental->forEachDTreeWrapperParallel([](DTreeWrapper* dTree) { dTree->buildpost(); });
        // }


#ifdef DEBUG_DISTR
        m_sdTree->my_vis();
#endif
    }

    void dumpSDTree(Scene* scene, ref<Sensor> sensor) {
        std::ostringstream extension;
        extension << "-" << std::setfill('0') << std::setw(2) << m_iter << ".sdt";
        fs::path path = scene->getDestinationFile();
        //path = path.parent_path() / (path.leaf().string() + extension.str());

        auto cameraMatrix = sensor->getWorldTransform()->eval(0).getMatrix();

        BlobWriter blob(path.string());

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                blob << (float)cameraMatrix(i, j);
            }
        }

        m_sdTree->dump(blob);
    }

    bool performRenderPasses(Float& variance, int numPasses, Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID, int integratorResID) {

        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        int repeatCount = 1;

        auto start = std::chrono::steady_clock::now();

        for (int r = 0; r < repeatCount; ++r) {
            Log(EInfo, "Rendering %d render passes.", numPasses);

            m_image->clear();
            m_squaredImage->clear();

            HDTimer timer_phase_renderpass;

            for (int i = 0; i < numPasses; ++i) {
                ref<BlockedRenderProcess> process = renderPass(scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID);
                m_renderProcesses.push_back(process);
            }

            bool result = true;
            int passesRenderedLocal = 0;

            static const size_t processBatchSize = 128;

            for (size_t i = 0; i < m_renderProcesses.size(); i += processBatchSize) {
                const size_t start = i;
                const size_t end = std::min(i + processBatchSize, m_renderProcesses.size());
                for (size_t j = start; j < end; ++j) {
                    sched->schedule(m_renderProcesses[j]);
                }

                for (size_t j = start; j < end; ++j) {
                    auto& process = m_renderProcesses[j];
                    sched->wait(process);

                    ++m_passesRendered;
                    ++m_passesRenderedThisIter;
                    ++passesRenderedLocal;

                    int progress = 0;
                    bool shouldAbort;
                    switch (m_budgetType) {
                    case ESpp:
                        progress = m_passesRendered;
                        shouldAbort = false;
                        break;
                    case ESeconds:
                        progress = (int)computeElapsedSeconds(m_startTime);
                        shouldAbort = progress > m_budget;
                        break;
                    default:
                        Assert(false);
                        break;
                    }

                    m_progress->update(progress);

                    if (process->getReturnStatus() != ParallelProcess::ESuccess) {
                        result = false;
                        shouldAbort = true;
                    }

                    if (shouldAbort) {
                        goto l_abort;
                    }
                }
            }

            statsPhaseTimeRenderPass += timer_phase_renderpass.value();

            m_renderProcesses.clear();
        }

        return true;

    l_abort:

        for (auto& process : m_renderProcesses) {
            sched->cancel(process);
        }

        Float seconds = computeElapsedSeconds(start);

        // Log(EInfo, "%.2f seconds, Total passes: %d",
        //     seconds, m_passesRendered);

        return false;
    }

    bool doNeeWithSpp(int spp) {
        switch (m_nee) {
        case ENever:
            return false;
        case EKickstart:
            return spp < 128;
        default:
            return true;
        }
    }

    bool renderSPP(Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID, int integratorResID) {

        ref<Scheduler> sched = Scheduler::getInstance();

        size_t sampleCount = (size_t)m_budget;

        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();
        // getResource(): tried to look up multi resource 2 without specifying a core index!
        //Sampler* sampler = static_cast<Sampler*>(sched->getResource(samplerResID, 0));
        const int nSamplers = sched->getCoreCount();
        std::vector<ref<Sampler>> samplers(nSamplers); // to splat samples
        for (int i = 0; i < nSamplers; ++i) {
            Properties props("deterministic");
            props.setSize("sampleCount", 99999999);
            props.setSize("seed", i);
            samplers[i] = static_cast<Sampler*>(PluginManager::getInstance()->createObject(MTS_CLASS(Sampler), props));
        }

        bool result = true;

        int nPasses = (int)std::ceil(sampleCount / (Float)m_sppPerPass);
        // nPasses += (nPasses & 1);
        sampleCount = m_sppPerPass * nPasses;

        Float currentVarAtEnd = std::numeric_limits<Float>::infinity();

        m_progress = std::unique_ptr<ProgressReporter>(new ProgressReporter("Rendering", nPasses, job));

        while (result && m_passesRendered < nPasses) {
            g_iter = m_iter;

            HDTimer timer_phase_total;

            const int sppRendered = m_passesRendered * m_sppPerPass;
            m_doNee = doNeeWithSpp(sppRendered);

            int remainingPasses = nPasses - m_passesRendered;

            int proposalPassesThisIter = 0;
            if (m_workingMode == 1) {
                proposalPassesThisIter = 1;
            }

            int passesThisIteration = std::min(remainingPasses, proposalPassesThisIter);

            // If the next iteration does not manage to double the number of passes once more
            // then it would be unwise to throw away the current iteration. Instead, extend
            // the current iteration to the end.
            // This condition can also be interpreted as: the last iteration must always use
            // at _least_ half the total sample budget.
            float fac = 2.0f;
            if (remainingPasses - passesThisIteration < fac * passesThisIteration) {
                if (m_sampleAllocSeq != ESampleAllocSeq::EUniform) {
                    passesThisIteration = remainingPasses;
                }
            }

            m_passThisIteration = passesThisIteration;
            g_passesThisIter = passesThisIteration;

            m_isFinalIter = passesThisIteration >= remainingPasses;

            film->clear();

            const auto iterBaseTime = std::chrono::steady_clock::now();

            HDTimer export_timer;

            if (m_workingMode != 0) {
                if (m_useMixSDTree) {
                    OFSTREAM ofs("guiding_log_mix.txt");
                    m_sdTreeMix->my_export(ofs);
                    ofs.close();
                }
                else if (m_useSDTree_SRB5) {
                    OFSTREAM ofs("guiding_log_mix.txt");
                    m_sdTreeSRB5->my_export(ofs);
                    ofs.close();
                }
               
            }

            static float total_export_time = 0;
            total_export_time += export_timer.value();
            printf("    exportsdt uses %.3f sec, total %.3f\n", export_timer.value(), total_export_time);


            HDTimer timer_phase_rendering;

            // give the turn to mts3
            remove("guiding_turn1.txt");
            std::ofstream ofs("guiding_turn3.txt");
            ofs << " ";
            ofs.close();

            // waiting for mts3 to finish
            while (true) {
                std::ifstream ifse("guiding_turn1end.txt");
                if (ifse.good()) {
                    exit(0);
                }
                std::ifstream ifs("guiding_turn1.txt");
                if (ifs.good()) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            remove("guiding_turn1.txt");

            g_passesDone += g_passesThisIter;

            // if (global_decay_rate == 0) {
            //     initSDTree();
            //     for (int x=0;x<3;x++) {
            //         resetSDTree();
            //         commitRecordedSamples(sampler);
            //         buildSDTree();
            //     }
            // }
            resetSDTree();
            commitRecordedSamples(samplers);
            buildSDTree();

            if (m_dumpSDTree && !m_isFinalIter) {
                dumpSDTree(scene, sensor);
            }

            ++m_iter;
            m_passesRenderedThisIter = 0;

            statsPhaseTimeTotal += timer_phase_total.value();

            printMystats();
            remove("guiding_turn1.txt");
        }

        return result;
    }

    bool renderTime(Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID, int integratorResID) {

        std::cout << "not supported" << std::endl;
        std::cerr << "not supported" << std::endl;
        exit(1);

        return false;
    }

    void commitRecordedSamples(std::vector<ref<Sampler>>& samplers) {
        float import_time = 0;
        float splat_time = 0;
        const auto spatialFilter_Primal = m_spatialFilter_Primal;  // not work for srb now
        const auto spatialFilter_Adjoint = m_spatialFilter_Adjoint; 

        if (m_useSDTree_SRB5) {

            HDTimer import_timer;
            auto samples = readFromFile("guiding_samples.txt"); // all samples
            import_time += import_timer.value();
            printf("    rdsamples uses %.3f sec, total %.3f sec\n", import_timer.value(), import_time);
            Assert(sizeof(double) == 8); // np.float64
            Assert(samples.size() % 64 == 0); // each line 8 number(the last one is tag), each 8 bytes
            int n = samples.size() / 64;
            double* ptr = reinterpret_cast<double*>(samples.data());

            HDTimer splat_timer;
#pragma omp parallel for
            for (int j = 0; j < n; ++j) {
                double buf[8];
                for (int i = 0; i < 8; ++i) buf[i] = ptr[j * 8 + i];
                Sampler* sampler = samplers[j % samplers.size()];

                const StepperRawSample sample = { Point3f(buf[0], buf[1], buf[2]), Vector3f(buf[3], buf[4], buf[5]), Float(buf[6]) };
                auto bsdfSamplingFractionLoss = EBsdfSamplingFractionLoss::ENone;
                auto directionalFilter = m_directionalFilter;
                DTreeRecord rec{ sample.dir, sample.val, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, false, sample.pos };

                Vector dTreeVoxelSize_primal;
                Vector dTreeVoxelSize_pos;
                Vector dTreeVoxelSize_neg;
                STree<STreeNode5>* tree = m_sdTreeSRB5.get();
                Assert(tree != nullptr);
                DTreeWrapper* dTreePrimal = nullptr;
                DTreeWrapper* dTreePos = nullptr;
                DTreeWrapper* dTreeNeg = nullptr;


                const Float tag = Float(buf[7]);  // should be one of [1,2,3]  1-Ldf 2-dLf 3-Lf
                const Float errorBound = 0.5;
                bool recordPrimal = false;
                EDTreeType dtreePosType, dtreeNegType;

                if (std::abs(tag - 1) < errorBound) {
                    dTreePos = tree->dTreeWrapper(sample.pos, dTreeVoxelSize_pos, EDTreeType::ELdf_Pos);
                    dTreeNeg = tree->dTreeWrapper(sample.pos, dTreeVoxelSize_neg, EDTreeType::ELdf_Neg);
                    dtreePosType = EDTreeType::ELdf_Pos;
                    dtreeNegType = EDTreeType::ELdf_Neg;
                }
                else if (std::abs(tag - 2) < errorBound) {
                    dTreePos = tree->dTreeWrapper(sample.pos, dTreeVoxelSize_pos, EDTreeType::EdLf_Pos);
                    dTreeNeg = tree->dTreeWrapper(sample.pos, dTreeVoxelSize_neg, EDTreeType::EdLf_Neg);
                    dtreePosType = EDTreeType::EdLf_Pos;
                    dtreeNegType = EDTreeType::EdLf_Neg;
                }
                else if (std::abs(tag - 3) < errorBound) {
                    recordPrimal = true;
                    dTreePrimal = tree->dTreeWrapper(sample.pos, dTreeVoxelSize_primal, EDTreeType::ELf);
                }
                else {
                    Log(ELogLevel::EError, "unexpected tag value %f", tag);
                    Assert(false);
                }

                DTreeWrapper* dTreeGrad1 = sample.val > 0 ? dTreePos : dTreeNeg;
                DTreeWrapper* dTreeGrad2 = sample.val > 0 ? dTreeNeg : dTreePos;
                if (sample.val < 0) rec.radiance *= -1;


                switch (m_spatialFilter) {
                case ESpatialFilter::ENearest:
                    if (recordPrimal) {
                        dTreePrimal->record(rec, directionalFilter, bsdfSamplingFractionLoss);
                    }
                    else {
                        dTreeGrad1->record(rec, directionalFilter, bsdfSamplingFractionLoss);
                        rec.radiance = 0;
                        dTreeGrad2->record(rec, directionalFilter, bsdfSamplingFractionLoss);
                    }
                    break;
                case ESpatialFilter::EBox:
                    if (recordPrimal) {
                        tree->record(sample.pos, dTreeVoxelSize_primal, rec, directionalFilter, bsdfSamplingFractionLoss, EDTreeType::ELf);
                    }
                    else {
                        tree->record(sample.pos,
                            sample.val > 0 ? dTreeVoxelSize_pos : dTreeVoxelSize_neg,
                            rec, directionalFilter, bsdfSamplingFractionLoss,
                            sample.val > 0 ? dtreePosType : dtreeNegType);
                        rec.radiance = 0;
                        tree->record(sample.pos,
                            sample.val > 0 ? dTreeVoxelSize_neg : dTreeVoxelSize_pos,
                            rec, directionalFilter, bsdfSamplingFractionLoss,
                            sample.val > 0 ? dtreeNegType : dtreePosType);
                    }
                    break;
                case ESpatialFilter::EStochasticBox:
                {
                    if (recordPrimal) {
                        Vector offset = dTreeVoxelSize_primal;
                        offset.x *= sampler->next1D() - 0.5f;
                        offset.y *= sampler->next1D() - 0.5f;
                        offset.z *= sampler->next1D() - 0.5f;

                        Point origin = tree->aabb().clip2(sample.pos + offset);
                        DTreeWrapper* splatDTree = tree->dTreeWrapper(origin, EDTreeType::ELf);
                        splatDTree->record(rec, directionalFilter, bsdfSamplingFractionLoss);
                    }
                    else {
                        Vector offset = sample.val > 0 ? dTreeVoxelSize_pos : dTreeVoxelSize_neg;
                        offset.x *= sampler->next1D() - 0.5f;
                        offset.y *= sampler->next1D() - 0.5f;
                        offset.z *= sampler->next1D() - 0.5f;

                        Point origin = tree->aabb().clip2(sample.pos + offset);
                        DTreeWrapper* splatDTree = tree->dTreeWrapper(origin, sample.val > 0 ? dtreePosType : dtreeNegType);
                        splatDTree->record(rec, directionalFilter, bsdfSamplingFractionLoss);

                        rec.radiance = 0;

                        offset = sample.val > 0 ? dTreeVoxelSize_neg : dTreeVoxelSize_pos;
                        offset.x *= sampler->next1D() - 0.5f;
                        offset.y *= sampler->next1D() - 0.5f;
                        offset.z *= sampler->next1D() - 0.5f;

                        origin = tree->aabb().clip2(sample.pos + offset);
                        splatDTree = tree->dTreeWrapper(origin, sample.val > 0 ? dtreeNegType : dtreePosType);
                        splatDTree->record(rec, directionalFilter, bsdfSamplingFractionLoss);
                    }
                    break;
                }

                }
            }
            splat_time += splat_timer.value();
        }
        else {

            {
                HDTimer import_timer;
                auto primal_samples = readFromFile("guiding_samples.txt");
                import_time += import_timer.value();
                Assert(sizeof(double) == 8); // np.float64
                Assert(primal_samples.size() % 56 == 0); // each line 7 number, each 8 bytes
                int n = primal_samples.size() / 56;
                double* ptr = reinterpret_cast<double*>(primal_samples.data());

                HDTimer splat_timer;
    #pragma omp parallel for
                for (int j = 0; j < n; ++j) {
                    double buf[7];
                    for (int i = 0; i < 7; ++i) buf[i] = ptr[j * 7 + i];
                    Sampler* sampler = samplers[j % samplers.size()];

                    const StepperRawSample sample = { Point3f(buf[0], buf[1], buf[2]), Vector3f(buf[3], buf[4], buf[5]), Float(buf[6]) };
                    auto bsdfSamplingFractionLoss = EBsdfSamplingFractionLoss::ENone;
                    auto directionalFilter = m_directionalFilter;
                    DTreeRecord rec{ sample.dir, sample.val, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, false, sample.pos };
                    //Vector dTreeVoxelSize;
                    //Vector dTreeVoxelSizehis;
                    //auto sdt = m_sdTree;
                    //auto sdthis = m_sdTreePrimalIncremental;

                    Vector dTreeVoxelSize_primal;
                    STree<STreeNode3PRB>* sdmix = m_useMixSDTree ? m_sdTreeMix.get() : nullptr;
                    DTreeWrapper* dTreePrimal = sdmix ? sdmix->dTreeWrapper(sample.pos, dTreeVoxelSize_primal, EDTreeType::EPrimal) : nullptr;

                    //DTreeWrapper* dTree = sdt->dTreeWrapper(sample.pos, dTreeVoxelSize);
                    //DTreeWrapper* dTreehis = sdthis->dTreeWrapper(sample.pos, dTreeVoxelSizehis);

                    switch (spatialFilter_Primal) {
                    case ESpatialFilter::ENearest:
                        //dTree->record(rec, directionalFilter, bsdfSamplingFractionLoss);
                        //dTreehis->record(rec, directionalFilter, bsdfSamplingFractionLoss);
                        if (dTreePrimal) {
                            dTreePrimal->record(rec, directionalFilter, bsdfSamplingFractionLoss);
                        }
                        break;
                    case ESpatialFilter::EBox:
                        //sdt->record(sample.pos, dTreeVoxelSize, rec, directionalFilter, bsdfSamplingFractionLoss);
                        //sdthis->record(sample.pos, dTreeVoxelSizehis, rec, directionalFilter, bsdfSamplingFractionLoss);
                        if (sdmix) {
                            sdmix->record(sample.pos, dTreeVoxelSize_primal, rec, directionalFilter, bsdfSamplingFractionLoss, EDTreeType::EPrimal);
                        }
                        break;
                    case ESpatialFilter::EStochasticBox:
                    {
                        if (sdmix) {
                            // Jitter the actual position within the
                            // filter box to perform stochastic filtering.
                            Vector offset = dTreeVoxelSize_primal;
                            offset.x *= sampler->next1D() - 0.5f;
                            offset.y *= sampler->next1D() - 0.5f;
                            offset.z *= sampler->next1D() - 0.5f;

                            Point origin = sdmix->aabb().clip2(sample.pos + offset);
                            DTreeWrapper* splatDTree = sdmix->dTreeWrapper(origin, EDTreeType::EPrimal);
                            splatDTree->record(rec, directionalFilter, bsdfSamplingFractionLoss);
                        }
                        break;
                    }

                    }
                }             
                splat_time += splat_timer.value();
            }

            if(m_workingMode == 1) {
                HDTimer import_timer;
                auto adjoint_samples = readFromFile("guiding_samples_adjoint.txt");
                import_time += import_timer.value();
                printf("    rdsamples uses %.3f sec, total %.3f sec\n", import_timer.value(), import_time);
                Assert(sizeof(double) == 8);
                Assert(adjoint_samples.size() % 56 == 0); // each line 7 number, each 8 bytes
                int n = adjoint_samples.size() / 56;
                double* ptr = reinterpret_cast<double*>(adjoint_samples.data());

                HDTimer splat_timer;
    #pragma omp parallel for

                for (int j = 0; j < n; ++j) {
                    double buf[7];
                    for (int i = 0; i < 7; ++i) buf[i] = ptr[j * 7 + i];
                    Sampler* sampler = samplers[j % samplers.size()];

                    const StepperRawSample sample = { Point3f(buf[0], buf[1], buf[2]), Vector3f(buf[3], buf[4], buf[5]), Float(buf[6]) };
                    auto bsdfSamplingFractionLoss = EBsdfSamplingFractionLoss::ENone;
                    auto directionalFilter = m_directionalFilter;
                    DTreeRecord rec{ sample.dir, sample.val, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, false, sample.pos };
                    //Vector dTreeVoxelSize;
                    //Vector dTreeVoxelSizehis;
                    //Vector dTreeVoxelSize_dec;
                    //Vector dTreeVoxelSize_dec2;

                    Vector dTreeVoxelSize_adpos;
                    Vector dTreeVoxelSize_adneg;
                    STree<STreeNode3PRB>* sdmix = m_useMixSDTree ? m_sdTreeMix.get() : nullptr;

                    DTreeWrapper* dTreeAdPos = sdmix ? sdmix->dTreeWrapper(sample.pos, dTreeVoxelSize_adpos, EDTreeType::EAdPos) : nullptr;
                    DTreeWrapper* dTreeAdNeg = sdmix ? sdmix->dTreeWrapper(sample.pos, dTreeVoxelSize_adneg, EDTreeType::EAdNeg) : nullptr;
                    DTreeWrapper* dTree_ad1 = sample.val > 0 ? dTreeAdPos : dTreeAdNeg;
                    DTreeWrapper* dTree_ad2 = sample.val > 0 ? dTreeAdNeg : dTreeAdPos;


                    //auto sdt = m_sdTree2;
                    //auto sdthis = m_sdTreeAdjointIncremental;
                    //auto sdt_decompose = sample.val > 0 ? m_sdTreeAdjointPos : m_sdTreeAdjointNeg;
                    //auto sdt_decompose2 = sample.val > 0 ? m_sdTreeAdjointNeg : m_sdTreeAdjointPos;
                    if (sample.val < 0) rec.radiance *= -1;

                    //DTreeWrapper* dTree = sdt->dTreeWrapper(sample.pos, dTreeVoxelSize);
                    //DTreeWrapper* dTreehis = sdthis->dTreeWrapper(sample.pos, dTreeVoxelSizehis);
                    //DTreeWrapper* dTree_decompose = sdt_decompose->dTreeWrapper(sample.pos, dTreeVoxelSize_dec);
                    //DTreeWrapper* dTree_decompose2 = sdt_decompose2->dTreeWrapper(sample.pos, dTreeVoxelSize_dec2);

                    switch (spatialFilter_Adjoint) {
                    case ESpatialFilter::ENearest:
                        //dTree->record(rec, directionalFilter, bsdfSamplingFractionLoss);
                        //dTreehis->record(rec, directionalFilter, bsdfSamplingFractionLoss);
                        //dTree_decompose->record(rec, directionalFilter, bsdfSamplingFractionLoss);
                        if (dTree_ad1) dTree_ad1->record(rec, directionalFilter, bsdfSamplingFractionLoss);
                        rec.radiance = 0;
                        //dTree_decompose2->record(rec, directionalFilter, bsdfSamplingFractionLoss);
                        if (dTree_ad2) dTree_ad2->record(rec, directionalFilter, bsdfSamplingFractionLoss);
                        break;
                    case ESpatialFilter::EBox:
                        //sdt->record(sample.pos, dTreeVoxelSize, rec, directionalFilter, bsdfSamplingFractionLoss);
                        //sdthis->record(sample.pos, dTreeVoxelSizehis, rec, directionalFilter, bsdfSamplingFractionLoss);
                        //sdt_decompose->record(sample.pos, dTreeVoxelSize_dec, rec, directionalFilter, bsdfSamplingFractionLoss);
                        if (sdmix) {
                            sdmix->record(sample.pos,
                                sample.val > 0 ? dTreeVoxelSize_adpos : dTreeVoxelSize_adneg,
                                rec, directionalFilter, bsdfSamplingFractionLoss,
                                sample.val > 0 ? EDTreeType::EAdPos : EDTreeType::EAdNeg);
                        }
                        rec.radiance = 0;
                        //sdt_decompose2->record(sample.pos, dTreeVoxelSize_dec2, rec, directionalFilter, bsdfSamplingFractionLoss);
                        if (sdmix) {
                            sdmix->record(sample.pos,
                                sample.val > 0 ? dTreeVoxelSize_adneg : dTreeVoxelSize_adpos,
                                rec, directionalFilter, bsdfSamplingFractionLoss,
                                sample.val > 0 ? EDTreeType::EAdNeg : EDTreeType::EAdPos);
                        }
                        break;
                    case ESpatialFilter::EStochasticBox:
                    {
                        if (sdmix) {
                            Vector offset = sample.val > 0 ? dTreeVoxelSize_adpos : dTreeVoxelSize_adneg;
                            offset.x *= sampler->next1D() - 0.5f;
                            offset.y *= sampler->next1D() - 0.5f;
                            offset.z *= sampler->next1D() - 0.5f;

                            Point origin = sdmix->aabb().clip2(sample.pos + offset);
                            DTreeWrapper* splatDTree = sdmix->dTreeWrapper(origin, sample.val > 0 ? EDTreeType::EAdPos : EDTreeType::EAdNeg);
                            splatDTree->record(rec, directionalFilter, bsdfSamplingFractionLoss);

                            rec.radiance = 0;

                            offset = sample.val > 0 ? dTreeVoxelSize_adneg : dTreeVoxelSize_adpos;
                            offset.x *= sampler->next1D() - 0.5f;
                            offset.y *= sampler->next1D() - 0.5f;
                            offset.z *= sampler->next1D() - 0.5f;                       
                            
                            origin = sdmix->aabb().clip2(sample.pos + offset);
                            splatDTree = sdmix->dTreeWrapper(origin, sample.val > 0 ? EDTreeType::EAdNeg : EDTreeType::EAdPos);
                            splatDTree->record(rec, directionalFilter, bsdfSamplingFractionLoss);
                        }
                        break;
                    }

                    }
                }
                splat_time += splat_timer.value();
            }

        }


        static float total_import_time = 0;
        total_import_time += import_time;
        
        printf("    sdtcommit uses %.3f sec, total %.3f sec\n", splat_time);
    }

    bool render(Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID) {

        m_sdTree = std::make_shared<STree<STreeNode>>(m_aabb);
        if (m_workingMode == 1) {
           
            if (m_useMixSDTree) {
                m_sdTreeMix = std::make_shared<STree<STreeNode3PRB>>(m_aabb);
            }
            if (m_useSDTree_SRB5) {
                m_sdTreeSRB5 = std::make_shared<STree<STreeNode5>>(m_aabb);
            }
        }
        m_iter = 0;
        m_isFinalIter = false;

        ref<Scheduler> sched = Scheduler::getInstance();

        size_t nCores = sched->getCoreCount();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        m_film = film;

        auto properties = Properties("hdrfilm");
        properties.setInteger("width", film->getSize().x);
        properties.setInteger("height", film->getSize().y);
        m_varianceBuffer = static_cast<Film*>(PluginManager::getInstance()->createObject(MTS_CLASS(Film), properties));
        m_varianceBuffer->setDestinationFile(scene->getDestinationFile(), 0);

        m_squaredImage = new ImageBlock(Bitmap::ESpectrumAlphaWeight, film->getCropSize());
        m_image = new ImageBlock(Bitmap::ESpectrumAlphaWeight, film->getCropSize());

        m_images.clear();
        m_variances.clear();
        m_squaredImages.clear();
        m_sampleCounts.clear();

        Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y, nCores, nCores == 1 ? "core" : "cores");

        Thread::initializeOpenMP(nCores);

        int integratorResID = sched->registerResource(this);
        bool result = true;

        int sppPerPass = m_sppPerPass;


        m_startTime = std::chrono::steady_clock::now();

        m_passesRendered = 0;
        switch (m_budgetType) {
        case ESpp:
            result = renderSPP(scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID);
            break;
        case ESeconds:
            result = renderTime(scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID);
            break;
        default:
            Assert(false);
            break;
        }

        sched->unregisterResource(integratorResID);

        m_progress = nullptr;

        printMystats();

        return result;
    }

    void renderBlock(const Scene *scene, const Sensor *sensor,
        Sampler *sampler, ImageBlock *block, const bool &stop,
        const std::vector< TPoint2<uint8_t> > &points) const {

        HDTimer timer;

        Float diffScaleFactor = 1.0f /
            std::sqrt((Float)m_sppPerPass);

        bool needsApertureSample = sensor->needsApertureSample();
        bool needsTimeSample = sensor->needsTimeSample();

        RadianceQueryRecord rRec(scene, sampler);
        Point2 apertureSample(0.5f);
        Float timeSample = 0.5f;
        RayDifferential sensorRay;

        block->clear();

        ref<ImageBlock> squaredBlock = block->clone();

        uint32_t queryType = RadianceQueryRecord::ESensorRay;

        if (!sensor->getFilm()->hasAlpha()) // Don't compute an alpha channel if we don't have to
            queryType &= ~RadianceQueryRecord::EOpacity;

        for (size_t i = 0; i < points.size(); ++i) {
            Point2i offset = Point2i(points[i]) + Vector2i(block->getOffset());
            if (stop)
                break;

            for (int j = 0; j < m_sppPerPass; j++) {
                rRec.newQuery(queryType, sensor->getMedium());
                Point2 samplePos(Point2(offset) + Vector2(rRec.nextSample2D()));

                if (needsApertureSample)
                    apertureSample = rRec.nextSample2D();
                if (needsTimeSample)
                    timeSample = rRec.nextSample1D();

                Spectrum spec = sensor->sampleRayDifferential(
                    sensorRay, samplePos, apertureSample, timeSample);

                sensorRay.scaleDifferential(diffScaleFactor);

                rRec.samplePos = samplePos;
                rRec.baseSpec = spec;

                spec *= Li(sensorRay, rRec);
                block->put(samplePos, spec, rRec.alpha);
                squaredBlock->put(samplePos, spec * spec, rRec.alpha);

                sampler->advance();
            }
        }

        m_squaredImage->put(squaredBlock);
        m_image->put(block);

        statsPhaseTimeRenderBlockSum += timer.value();
    }

    void cancel() {
        const auto& scheduler = Scheduler::getInstance();
        for (size_t i = 0; i < m_renderProcesses.size(); ++i) {
            scheduler->cancel(m_renderProcesses[i]);
        }
    }

    Spectrum sampleMat(const BSDF* bsdf, BSDFSamplingRecord& bRec, Float& woPdf, Float& bsdfPdf, Float& dTreePdf, Float bsdfSamplingFraction, RadianceQueryRecord& rRec, const DTreeWrapper* dTree) const {
        Point2 sample = rRec.nextSample2D();

        auto type = bsdf->getType();
        if (!m_isBuilt || !dTree || (type & BSDF::EDelta) == (type & BSDF::EAll)) {
            auto result = bsdf->sample(bRec, bsdfPdf, sample);
            woPdf = bsdfPdf;
            dTreePdf = 0;
            return result;
        }

        HDTimer timer;

        Spectrum result;
        if (sample.x < bsdfSamplingFraction) {
            sample.x /= bsdfSamplingFraction;
            result = bsdf->sample(bRec, bsdfPdf, sample);
            if (result.isZero()) {
                woPdf = bsdfPdf = dTreePdf = 0;
                return Spectrum{0.0f};
            }

            // If we sampled a delta component, then we have a 0 probability
            // of sampling that direction via guiding, thus we can return early.
            if (bRec.sampledType & BSDF::EDelta) {
                dTreePdf = 0;
                woPdf = bsdfPdf * bsdfSamplingFraction;
                return result / bsdfSamplingFraction;
            }

            result *= bsdfPdf;
        } else {
            sample.x = (sample.x - bsdfSamplingFraction) / (1 - bsdfSamplingFraction);
            bRec.wo = bRec.its.toLocal(dTree->sample(rRec.sampler));
            result = bsdf->eval(bRec);
        }

        pdfMat(woPdf, bsdfPdf, dTreePdf, bsdfSamplingFraction, bsdf, bRec, dTree);

        statsPhaseTimeSampleMat += timer.value();

        if (woPdf == 0) {
            return Spectrum{0.0f};
        }

        return result / woPdf;
    }

    void pdfMat(Float& woPdf, Float& bsdfPdf, Float& dTreePdf, Float bsdfSamplingFraction, const BSDF* bsdf, const BSDFSamplingRecord& bRec, const DTreeWrapper* dTree) const {
        dTreePdf = 0;

        auto type = bsdf->getType();
        if (!m_isBuilt || !dTree || (type & BSDF::EDelta) == (type & BSDF::EAll)) {
            woPdf = bsdfPdf = bsdf->pdf(bRec);
            return;
        }

        bsdfPdf = bsdf->pdf(bRec);
        if (!std::isfinite(bsdfPdf)) {
            woPdf = 0;
            return;
        }

        dTreePdf = dTree->pdf(bRec.its.toWorld(bRec.wo));

        woPdf = bsdfSamplingFraction * bsdfPdf + (1 - bsdfSamplingFraction) * dTreePdf;
    }


    struct Vertex {
        DTreeWrapper* dTree;
        Vector dTreeVoxelSize;
        Ray ray;

        Spectrum throughput;
        Spectrum bsdfVal;

        Spectrum radiance;

        Float woPdf, bsdfPdf, dTreePdf;
        bool isDelta;

        void record(const Spectrum& r) {
            radiance += r;
        }

        //  not use this now
        void commit(STree<STreeNode>& sdTree, Float statisticalWeight, ESpatialFilter spatialFilter, EDirectionalFilter directionalFilter, EBsdfSamplingFractionLoss bsdfSamplingFractionLoss, Sampler* sampler) {
            if (!(woPdf > 0) || !radiance.isValid() || !bsdfVal.isValid()) {
                return;
            }

            statsCommitRequestTotal ++;

            HDTimer timer;
            Spectrum localRadiance = Spectrum{0.0f};
            if (throughput[0] * woPdf > Epsilon) localRadiance[0] = radiance[0] / throughput[0];
            if (throughput[1] * woPdf > Epsilon) localRadiance[1] = radiance[1] / throughput[1];
            if (throughput[2] * woPdf > Epsilon) localRadiance[2] = radiance[2] / throughput[2];
            Spectrum product = localRadiance * bsdfVal;

            DTreeRecord rec{ ray.d, localRadiance.average(), product.average(), woPdf, bsdfPdf, dTreePdf, statisticalWeight, isDelta };
            switch (spatialFilter) {
            case ESpatialFilter::ENearest:
                dTree->record(rec, directionalFilter, bsdfSamplingFractionLoss);
                break;
            case ESpatialFilter::EStochasticBox:
            {
                DTreeWrapper* splatDTree = dTree;

                // Jitter the actual position within the
                // filter box to perform stochastic filtering.
                Vector offset = dTreeVoxelSize;
                offset.x *= sampler->next1D() - 0.5f;
                offset.y *= sampler->next1D() - 0.5f;
                offset.z *= sampler->next1D() - 0.5f;

                Point origin = sdTree.aabb().clip2(ray.o + offset);
                splatDTree = sdTree.dTreeWrapper(origin);
                if (splatDTree) {
                    splatDTree->record(rec, directionalFilter, bsdfSamplingFractionLoss);
                }
                break;
            }
            case ESpatialFilter::EBox:
                sdTree.record(ray.o, dTreeVoxelSize, rec, directionalFilter, bsdfSamplingFractionLoss);
                break;
            }
            statsPhaseTimeCommit += timer.value();
            statsCommitCall ++;
        }
    };

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        std::vector <BSDFSamplingRecord> bRecList;
        auto samplePos = rRec.samplePos;
        auto baseSpec = rRec.baseSpec;


        static const int MAX_NUM_VERTICES = 32;
        std::array<Vertex, MAX_NUM_VERTICES> vertices;

       
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        MediumSamplingRecord mRec;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        Float eta = 1.0f;

       
        rRec.rayIntersect(ray);

        Spectrum throughput(1.0f);
        bool scattered = false;

        int nVertices = 0;

        // Currently, these two functions as used at exactly the same manner

        auto recordRadiance = [&](Spectrum radiance) {
            Li += radiance;
            for (int i = 0; i < nVertices; ++i) {
                vertices[i].record(radiance);
            }
            };

        bool isGlintyPath = true;


        while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {

           
           
           
            if (rRec.medium && rRec.medium->sampleDistance(Ray(ray, 0, its.t), mRec, rRec.sampler)) {
#ifdef ENABLE_MEDIUM
               
                const PhaseFunction *phase = mRec.getPhaseFunction();

                if (rRec.depth >= m_maxDepth && m_maxDepth != -1) // No more scattering events allowed
                    break;

                throughput *= mRec.sigmaS * mRec.transmittance / mRec.pdfSuccess;

               
               
               

               
                DirectSamplingRecord dRec(mRec.p, mRec.time);

                if (rRec.type & RadianceQueryRecord::EDirectMediumRadiance) {
                    int interactions = m_maxDepth - rRec.depth - 1;

                    Spectrum value = scene->sampleAttenuatedEmitterDirect(
                        dRec, rRec.medium, interactions,
                        rRec.nextSample2D(), rRec.sampler);

                    if (!value.isZero()) {
                        const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

                       
                        PhaseFunctionSamplingRecord pRec(mRec, -ray.d, dRec.d);
                        Float phaseVal = phase->eval(pRec);

                        if (phaseVal != 0) {
                           
                            Float phasePdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                                ? phase->pdf(pRec) : (Float) 0.0f;

                           
                            const Float weight = miWeight(dRec.pdf, phasePdf);
                            recordRadiance(throughput * value * phaseVal * weight);
                        }
                    }
                }

               
               
               

                Float phasePdf;
                PhaseFunctionSamplingRecord pRec(mRec, -ray.d);
                Float phaseVal = phase->sample(pRec, phasePdf, rRec.sampler);
                if (phaseVal == 0)
                    break;
                throughput *= phaseVal;

               
                ray = Ray(mRec.p, pRec.wo, ray.time);
                ray.mint = 0;

                Spectrum value(0.0f);
                rayIntersectAndLookForEmitter(scene, rRec.sampler, rRec.medium,
                    m_maxDepth - rRec.depth - 1, ray, its, dRec, value);

               
                if (!value.isZero() && (rRec.type & RadianceQueryRecord::EDirectMediumRadiance)) {
                    const Float emitterPdf = scene->pdfEmitterDirect(dRec);
                    recordRadiance(throughput * value * miWeight(phasePdf, emitterPdf));
                }

               
               
               

               
                if (!(rRec.type & RadianceQueryRecord::EIndirectMediumRadiance))
                    break;
                rRec.type = RadianceQueryRecord::ERadianceNoEmission;

                if (rRec.depth++ >= m_rrDepth) {
                   

                    Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
                    if (rRec.nextSample1D() >= q)
                        break;
                    throughput /= q;
                }
#else
                break;
#endif

                /////////////////////////////////////////////////////////////////////////////////////////////////

            } else {
               
               
               

               
                if (rRec.medium)
                    throughput *= mRec.transmittance / mRec.pdfFailure;

                if (!its.isValid()) {
                   
                    if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
                        && (!m_hideEmitters || scattered)) {
                        Spectrum value = scene->evalEnvironment(ray);
                        if (rRec.medium)
                            value *= rRec.medium->evalTransmittance(ray, rRec.sampler);
                        // * record
                        recordRadiance(throughput * value);
                    }

                    break;
                }

               
                if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
                    && (!m_hideEmitters || scattered)) {
                    // * record
                    recordRadiance(throughput * its.Le(-ray.d));
                }

                //
                // if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance)) {
                //     recordRadiance(throughput * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth));
                // }

                if (rRec.depth >= m_maxDepth && m_maxDepth != -1)
                    break;

               
                Float wiDotGeoN = -dot(its.geoFrame.n, ray.d),
                    wiDotShN = Frame::cosTheta(its.wi);
                if (wiDotGeoN * wiDotShN < 0 && m_strictNormals)
                    break;

                const BSDF *bsdf = its.getBSDF();

                Vector dTreeVoxelSize;
                DTreeWrapper* dTree = nullptr;

                // We only guide smooth BRDFs for now. Analytic product sampling
                // would be conceivable for discrete decisions such as refraction vs
                // reflection.
                if (bsdf->getType() & BSDF::ESmooth) {
                    dTree = m_sdTree->dTreeWrapper(its.p, dTreeVoxelSize);
                }

                Float bsdfSamplingFraction = m_bsdfSamplingFraction;
                if (dTree && m_bsdfSamplingFractionLoss != EBsdfSamplingFractionLoss::ENone) {
                    bsdfSamplingFraction = dTree->bsdfSamplingFraction();
                }

               
               
               

               
                BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
                Float woPdf, bsdfPdf, dTreePdf;
                Spectrum bsdfWeight = sampleMat(bsdf, bRec, woPdf, bsdfPdf, dTreePdf, bsdfSamplingFraction, rRec, dTree);

                Float woPdf_film = 0;



                woPdf_film = woPdf;

                bool isDelta = bRec.sampledType & BSDF::EDelta;
                if (isDelta == false) {
                    isGlintyPath = false;
                }

               
               
               

                DirectSamplingRecord dRec(its);

#ifdef ENABLE_NEE
               
                if (m_doNee &&
                    (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) &&
                    (bsdf->getType() & BSDF::ESmooth)) {
                    int interactions = m_maxDepth - rRec.depth - 1;

                    Spectrum value = scene->sampleAttenuatedEmitterDirect(
                        dRec, its, rRec.medium, interactions,
                        rRec.nextSample2D(), rRec.sampler);

                    if (!value.isZero()) {
                        BSDFSamplingRecord bRec(its, its.toLocal(dRec.d));

                        Float woDotGeoN = dot(its.geoFrame.n, dRec.d);

                       
                        if (!m_strictNormals || woDotGeoN * Frame::cosTheta(bRec.wo) > 0) {
                           
                            const Spectrum bsdfVal = bsdf->eval(bRec);

                           
                            const Emitter *emitter = static_cast<const Emitter *>(dRec.object);
                            Float woPdf = 0, bsdfPdf = 0, dTreePdf = 0;
                            if (emitter->isOnSurface() && dRec.measure == ESolidAngle) {
                                pdfMat(woPdf, bsdfPdf, dTreePdf, bsdfSamplingFraction, bsdf, bRec, dTree);
                            }

                           
                            const Float weight = miWeight(dRec.pdf, woPdf);

                            value *= bsdfVal;
                            Spectrum L = throughput * value * weight;

                            if (!m_isFinalIter && m_nee != EAlways) {
                                if (dTree) {
                                    Vertex v = Vertex{
                                        dTree,
                                        dTreeVoxelSize,
                                        Ray(its.p, dRec.d, 0),
                                        throughput * bsdfVal / dRec.pdf,
                                        bsdfVal,
                                        L,
                                        dRec.pdf,
                                        bsdfPdf,
                                        dTreePdf,
                                        false,
                                    };

                                    v.commit(*m_sdTree, 0.5f, m_spatialFilter, m_directionalFilter, m_isBuilt ? m_bsdfSamplingFractionLoss : EBsdfSamplingFractionLoss::ENone, rRec.sampler);
                                }
                            }

                            recordRadiance(L);
                        }
                    }
                }
#endif

                // BSDF handling
                if (bsdfWeight.isZero())
                    break;

               
                const Vector wo = its.toWorld(bRec.wo);
                Float woDotGeoN = dot(its.geoFrame.n, wo);

                if (woDotGeoN * Frame::cosTheta(bRec.wo) <= 0 && m_strictNormals)
                    break;

               
                ray = Ray(its.p, wo, ray.time);

               
                throughput *= bsdfWeight;

                eta *= bRec.eta;
                if (its.isMediumTransition())
                    rRec.medium = its.getTargetMedium(ray.d);

               
                if (bRec.sampledType == BSDF::ENull) {
                    if (!(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                        break;

                    // There exist materials that are smooth/null hybrids (e.g. the mask BSDF), which means that
                    // for optimal-sampling-fraction optimization we need to record null transitions for such BSDFs.
                    if (m_bsdfSamplingFractionLoss != EBsdfSamplingFractionLoss::ENone &&
                        dTree && nVertices < MAX_NUM_VERTICES && !m_isFinalIter) {

                        if (1 / woPdf > 0) {
                            vertices[nVertices] = Vertex{
                                dTree,
                                dTreeVoxelSize,
                                ray,
                                throughput,
                                bsdfWeight * woPdf,
                                Spectrum{0.0f},
                                woPdf,
                                bsdfPdf,
                                dTreePdf,
                                true,
                            };

                            ++nVertices;
                        }
                    }

                    rRec.type = scattered ? RadianceQueryRecord::ERadianceNoEmission
                        : RadianceQueryRecord::ERadiance;
                    scene->rayIntersect(ray, its);
                    rRec.depth++;
                    continue;
                }

                Spectrum value(0.0f);
                rayIntersectAndLookForEmitter(scene, rRec.sampler, rRec.medium,
                    m_maxDepth - rRec.depth - 1, ray, its, dRec, value);

               
                // * We currently disable NEE, so need no care here
                if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) {
                    bool isDelta = bRec.sampledType & BSDF::EDelta;
                    const Float emitterPdf = (m_doNee && !isDelta && !value.isZero()) ? scene->pdfEmitterDirect(dRec) : 0;

                    const Float weight = miWeight(woPdf, emitterPdf);
                    Spectrum L = throughput * value * weight;
                    if (!L.isZero() && (!isGlintyPath || !m_filteredGlints)) {
                        recordRadiance(L);
                    }

                    if ((!isDelta || m_bsdfSamplingFractionLoss != EBsdfSamplingFractionLoss::ENone) &&
                        dTree && nVertices < MAX_NUM_VERTICES && !m_isFinalIter) {
                        if (1 / woPdf > 0) {
                            vertices[nVertices] = Vertex{
                                dTree,
                                dTreeVoxelSize,
                                ray,
                                throughput,
                                bsdfWeight * woPdf,
                                (m_nee == EAlways) ? Spectrum{0.0f} : L,
                                woPdf,
                                bsdfPdf,
                                dTreePdf,
                                isDelta,
                            };

                            ++nVertices;
                        }
                    }
                }

               
               
               

               
                if (!(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                    break;

                rRec.type = RadianceQueryRecord::ERadianceNoEmission;

                // Russian roulette
                if (rRec.depth++ >= m_rrDepth) {
                    Float successProb = 1.0f;
                    if (dTree && !(bRec.sampledType & BSDF::EDelta)) {
                        if (!m_isBuilt) {
                            successProb = throughput.max() * eta * eta;
                        } else {
                            // The adjoint russian roulette implementation of Mueller et al. [2017]
                            // was broken, effectively turning off russian roulette entirely.
                            // For reproducibility's sake, we therefore removed adjoint russian roulette
                            // from this codebase rather than fixing it.
                        }

                        successProb = std::max(0.1f, std::min(successProb, 0.99f));
                    }

                    if (rRec.nextSample1D() >= successProb)
                        break;
                    throughput /= successProb;
                }
            }

            scattered = true;
        }
        avgPathLength.incrementBase();
        avgPathLength += rRec.depth;

        if (nVertices > 0 && !m_isFinalIter && m_trainingFusion == 0) {
            for (int i = 0; i < nVertices; ++i) {
                vertices[i].commit(*m_sdTree,
                    m_nee == EKickstart && m_doNee ? 0.5f : 1.0f,
                    m_spatialFilter, m_directionalFilter,
                    m_isBuilt ? m_bsdfSamplingFractionLoss : EBsdfSamplingFractionLoss::ENone,
                    rRec.sampler);
            }
        }

        return Li;
    }

    /**
    * This function is called by the recursive ray tracing above after
    * having sampled a direction from a BSDF/phase function. Due to the
    * way in which this integrator deals with index-matched boundaries,
    * it is necessarily a bit complicated (though the improved performance
    * easily pays for the extra effort).
    *
    * This function
    *
    * 1. Intersects 'ray' against the scene geometry and returns the
    *    *first* intersection via the '_its' argument.
    *
    * 2. It checks whether the intersected shape was an emitter, or if
    *    the ray intersects nothing and there is an environment emitter.
    *    In this case, it returns the attenuated emittance, as well as
    *    a DirectSamplingRecord that can be used to query the hypothetical
    *    sampling density at the emitter.
    *
    * 3. If current shape is an index-matched medium transition, the
    *    integrator keeps on looking on whether a light source eventually
    *    follows after a potential chain of index-matched medium transitions,
    *    while respecting the specified 'maxDepth' limits. It then returns
    *    the attenuated emittance of this light source, while accounting for
    *    all attenuation that occurs on the wya.
    */
    void rayIntersectAndLookForEmitter(const Scene *scene, Sampler *sampler,
        const Medium *medium, int maxInteractions, Ray ray, Intersection &_its,
        DirectSamplingRecord &dRec, Spectrum &value) const {
        Intersection its2, *its = &_its;
        Spectrum transmittance(1.0f);
        bool surface = false;
        int interactions = 0;

        while (true) {
            surface = scene->rayIntersect(ray, *its);

            if (medium)
                transmittance *= medium->evalTransmittance(Ray(ray, 0, its->t), sampler);

            if (surface && (interactions == maxInteractions ||
                !(its->getBSDF()->getType() & BSDF::ENull) ||
                its->isEmitter())) {
               
                break;
            }

            if (!surface)
                break;

            if (transmittance.isZero())
                return;

            if (its->isMediumTransition())
                medium = its->getTargetMedium(ray.d);

            Vector wo = its->shFrame.toLocal(ray.d);
            BSDFSamplingRecord bRec(*its, -wo, wo, ERadiance);
            bRec.typeMask = BSDF::ENull;
            transmittance *= its->getBSDF()->eval(bRec, EDiscrete);

            ray.o = ray(its->t);
            ray.mint = Epsilon;
            its = &its2;

            if (++interactions > 100) { /// Just a precaution..
                Log(EWarn, "rayIntersectAndLookForEmitter(): round-off error issues?");
                return;
            }
        }

        if (surface) {
           
            if (its->isEmitter()) {
                dRec.setQuery(ray, *its);
                value = transmittance * its->Le(-ray.d);
            }
        } else {
           
            const Emitter *env = scene->getEnvironmentEmitter();

            if (env && env->fillDirectSamplingRecord(dRec, ray)) {
                value = transmittance * env->evalEnvironment(RayDifferential(ray));
                dRec.dist = std::numeric_limits<Float>::infinity();
                its->t = std::numeric_limits<Float>::infinity();
            }
        }
    }

    Float miWeight(Float pdfA, Float pdfB) const {
        pdfA *= pdfA; pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "GuidedPathTracer[" << endl
            << "  maxDepth = " << m_maxDepth << "," << endl
            << "  rrDepth = " << m_rrDepth << "," << endl
            << "  strictNormals = " << m_strictNormals << endl
            << "]";
        return oss.str();
    }

private:
    /// The datastructure for guiding paths.
    std::shared_ptr<STree<STreeNode>> m_sdTree;
    //std::shared_ptr<STree<STreeNode>> m_sdTree2;
    //std::shared_ptr<STree<STreeNode>> m_sdTreePrimalIncremental;
    //std::shared_ptr<STree<STreeNode>> m_sdTreeAdjointIncremental;
    //std::shared_ptr<STree<STreeNode>> m_sdTreeAdjointPos;  // positive adjoint(or gradient?)
    //std::shared_ptr<STree<STreeNode>> m_sdTreeAdjointNeg;  // negative
    //bool m_decomposeAdjoint;

    bool m_useMixSDTree;
    std::shared_ptr<STree<STreeNode3PRB>> m_sdTreeMix;

    bool m_useSDTree_SRB5;
    std::shared_ptr<STree<STreeNode5>> m_sdTreeSRB5;

    /// Record history distributions
    mutable std::vector<STree<STreeNode>> m_sdTreeList;
    /// The first phase of two-stage hack
    bool m_isPreStage = true;

    /// The squared values of our currently rendered image. Used to estimate variance.
    mutable ref<ImageBlock> m_squaredImage;
    /// The currently rendered image. Used to estimate variance.
    mutable ref<ImageBlock> m_image;

    std::vector<ref<Bitmap>> m_images;
    std::vector<Float> m_variances;
    std::vector<ref<Bitmap>> m_squaredImages;
    std::vector<int> m_sampleCounts;

    int m_spatialFilteringRepeat;
    Float m_spatialFilteringRadius;
    Float m_temporalFilteringWeight;
    int m_adjointTarget;
    mutable float m_tempParam;
    /// This contains the currently estimated variance.
    mutable ref<Film> m_varianceBuffer;

    /// The modes of NEE which are supported.
    enum ENee {
        ENever,
        EKickstart,
        EAlways,
    };

    /**
        How to perform next event estimation (NEE). The following values are valid:
        - "never":     Never performs NEE.
        - "kickstart": Performs NEE for the first few iterations to initialize
                       the SDTree with good direct illumination estimates.
        - "always":    Always performs NEE.
        Default = "never"
    */
    std::string m_neeStr;
    ENee m_nee;

    /// Whether Li should currently perform NEE (automatically set during rendering based on m_nee).
    bool m_doNee;

    enum EBudget {
        ESpp,
        ESeconds,
    };

    /**
        What type of budget to use. The following values are valid:
        - "spp":     Budget is the number of samples per pixel.
        - "seconds": Budget is a time in seconds.
        Default = "seconds"
    */
    std::string m_budgetStr;
    EBudget m_budgetType;
    Float m_budget;

    bool m_isBuilt = false;
    int m_iter;
    bool m_isFinalIter = false;
    int m_passThisIteration;

    int m_sppPerPass;

    int m_passesRendered;
    int m_passesRenderedThisIter;
    mutable std::unique_ptr<ProgressReporter> m_progress;

    std::vector<ref<BlockedRenderProcess>> m_renderProcesses;

    /**
        How to combine the samples from all path-guiding iterations:
        - "discard":    Discard all but the last iteration.
        - "automatic":  Discard all but the last iteration, but automatically assign an appropriately
                        larger budget to the last [Mueller et al. 2018].
        - "inversevar": Combine samples of the last 4 iterations based on their
                        mean pixel variance [Mueller et al. 2018].
        Default     = "automatic" (for reproducibility)
        Recommended = "inversevar"
    */
    std::string m_sampleCombinationStr;
    ESampleCombination m_sampleCombination;


    std::string m_sampleAllocSeqStr;
    ESampleAllocSeq m_sampleAllocSeq;


    /// Maximum memory footprint of the SDTree in MB. Stops subdividing once reached. -1 to disable.
    int m_sdTreeMaxMemory;

    /**
        The spatial filter to use when splatting radiance samples into the SDTree.
        The following values are valid:
        - "nearest":    No filtering [Mueller et al. 2017].
        - "stochastic": Stochastic box filter; improves upon Mueller et al. [2017]
                        at nearly no computational cost.
        - "box":        Box filter; improves the quality further at significant
                        additional computational cost.
        Default     = "nearest" (for reproducibility)
        Recommended = "stochastic"
    */
    std::string m_spatialFilterStr;
    ESpatialFilter m_spatialFilter;
    ESpatialFilter m_spatialFilter_Primal;
    ESpatialFilter m_spatialFilter_Adjoint; 

    /**
        The directional filter to use when splatting radiance samples into the SDTree.
        The following values are valid:
        - "nearest":    No filtering [Mueller et al. 2017].
        - "box":        Box filter; improves upon Mueller et al. [2017]
                        at nearly no computational cost.
        Default     = "nearest" (for reproducibility)
        Recommended = "box"
    */
    std::string m_directionalFilterStr;
    EDirectionalFilter m_directionalFilter;

    /**
        Leaf nodes of the spatial binary tree are subdivided if the number of samples
        they received in the last iteration exceeds c * sqrt(2^k) where c is this value
        and k is the iteration index. The first iteration has k==0.
        Default     = 12000 (for reproducibility)
        Recommended = 4000
    */
    int m_sTreeThreshold;

    /**
        Leaf nodes of the directional quadtree are subdivided if the fraction
        of energy they carry exceeds this value.
        Default = 0.01 (1%)
    */
    Float m_dTreeThreshold;

    /**
        When guiding, we perform MIS with the balance heuristic between the guiding
        distribution and the BSDF, combined with probabilistically choosing one of the
        two sampling methods. This factor controls how often the BSDF is sampled
        vs. how often the guiding distribution is sampled.
        Default = 0.5 (50%)
    */
    Float m_bsdfSamplingFraction;

    /**
        The loss function to use when learning the bsdfSamplingFraction using gradient
        descent, following the theory of Neural Importance Sampling [Mueller et al. 2018].
        The following values are valid:
        - "none":  No learning (uses the fixed `m_bsdfSamplingFraction`).
        - "kl":    Optimizes bsdfSamplingFraction w.r.t. the KL divergence.
        - "var":   Optimizes bsdfSamplingFraction w.r.t. variance.
        Default     = "none" (for reproducibility)
        Recommended = "kl"
    */
    std::string m_bsdfSamplingFractionLossStr;
    EBsdfSamplingFractionLoss m_bsdfSamplingFractionLoss;

    /**
        Whether to dump a binary representation of the SD-Tree to disk after every
        iteration. The dumped SD-Tree can be visualized with the accompanying
        visualizer tool.
        Default = false
    */
    bool m_dumpSDTree;

    bool m_lazyRebuild;
    bool m_freqRebuild;

    bool m_filteredGlints;

    bool m_trainingFusion;

    int m_workingMode; // = 1 if working with mts3 for dr

    /// The time at which rendering started.
    std::chrono::steady_clock::time_point m_startTime;
    ref<Film> m_film;

    AABB m_aabb;

public:
    MTS_DECLARE_CLASS()

        static std::mutex m_raw_buffer_lock;

};


MTS_IMPLEMENT_CLASS(GuidedPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(GuidedPathTracer, "Guided path tracer");
MTS_NAMESPACE_END
