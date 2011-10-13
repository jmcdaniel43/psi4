#ifndef _yeti_runtime_h
#define _yeti_runtime_h

#include <iostream>
#include <map>

#include "class.h"
#include "index.hpp"
#include "thread.hpp"
#include "data.hpp"
#include "cache.hpp"
#include "aobasis.hpp"
#include "messenger.hpp"

#ifdef redefine_size_t
#define size_t custom_size_t
#endif

namespace yeti {

class YetiOStream {

    public:
        template <typename T>
        void print(const T& t);


};


class TensorTest;
class YetiRuntime {

    public:
        typedef enum {Local, MPI} runtime_parallel_type_t;

    private:
        static std::map<std::string, IndexDescrPtr> descrs_;

        static std::map<IndexRange*, std::map<IndexRange*, int> > valid_subranges_;

        static std::map<std::string, AOBasisPtr> basis_sets_;

        static uli descr_id_count_;

        static uli* max_range_sizes_;

        static uli* max_block_sizes_;

        static usi max_depth_;

        static uli nthread_compute_;

        static uli nthread_comm_;

        static uli nthread_;

        static uli nproc_;

        static uli nblocks_to_allocate_;

        static size_t memory_;

        static size_t amount_mem_allocated_;

        static uli me_;

        static Thread** threads_;

        static bool threaded_compute_;

        static ThreadGroup* threadgrp_;

        static ThreadLock* printlock_;

        static ThreadLock* malloc_lock_;

        static void init_sizes();

        static void init_malloc();

        static usi print_depth_;

        static char* thread_stack_;

        static size_t thread_stack_size_;

        static void init_local_messenger();

        static void init_mpi_messenger();

        static void init_mpi();

        static runtime_parallel_type_t parallel_type_;

        static YetiMPIMessenger* mpi_messenger_;

        static YetiMessenger* messenger_;

    public:

        static IndexRange* get_range(const std::string& id);

        static IndexDescr* get_descr(const std::string& id);

        /**
          Configure system-dependent variables like the number
          of threads, total available memory, MPI environment
          @param me The processor number
          @param
        */
        static void init_system_environment();

        /**
          Configure molecule-dependent variables like max block size
          and scratch space which depend on particular molecule
        */
        static void init_molecule_environment();

        static void finalize();

        static void init_threads();

        static void reconfigure(uli me, uli nproc);

        static const uli* max_block_sizes();

        static usi max_depth();

        static const uli* max_range_sizes();

        static uli me();

        static size_t memory();

        static size_t cache_mem();

        static float matrix_multiply_cutoff;
        
        static float print_cutoff;
        
        static float nonnull_cutoff;

        static uli nproc();

        static uli get_thread_number();

#if USE_DEFAULT_THREAD_STACK
        static void set_thread_stack(uli threadnum, void* addr);
#endif

        static void register_index_range(
            IndexRange *range,
            const std::string& descr,
            const char* id1,
            const char* id2 = 0,
            const char* id3 = 0,
            const char* id4 = 0,
            const char* id5 = 0,
            const char* id6 = 0,
            const char* id7 = 0,
            const char* id8 = 0,
            const char* id9 = 0,
            const char* id10 = 0,
            const char* id11 = 0,
            const char* id12 = 0
        );

        static void register_index_descr(
            const IndexDescrPtr& descr,
            const char* id1,
            const char* id2 = 0,
            const char* id3 = 0,
            const char* id4 = 0,
            const char* id5 = 0,
            const char* id6 = 0,
            const char* id7 = 0,
            const char* id8 = 0,
            const char* id9 = 0,
            const char* id10 = 0,
            const char* id11 = 0,
            const char* id12 = 0
        );

        static void register_index_descr(
            const std::string& id,
            const IndexDescrPtr& descr
        );

        static void register_index_range(
            const IndexRangePtr& range,
            const std::string& name,
            const std::string& id

        );

        static uli nthread_compute();
        
        static uli nthread_comm();

        static uli nthread();

        static void add_thread(Thread* thr);

        static ThreadGroup* get_thread_grp();

        static void lock_print();

        static void unlock_print();

        static void lock_malloc();

        static void unlock_malloc();

        static usi print_depth();

        static void set_print_depth(usi depth);

        static size_t get_thread_stack_size();

        static void* get_thread_stack(uli threadnum);
        
        static void register_basis_set(const AOBasisPtr& basis);

        static const AOBasisPtr& get_basis(const std::string& id);

        static void register_subranges(
            IndexRange* parent,
            IndexRange* subrange1,
            IndexRange* subrange2 = 0,
            IndexRange* subrange3 = 0,
            IndexRange* subrange4 = 0,
            IndexRange* subrange5 = 0,
            IndexRange* subrange6 = 0
        );

        static void register_subranges(
            const char* parent,
            const char* subrange1,
            const char* subrange2 = 0,
            const char* subrange3 = 0,
            const char* subrange4 = 0,
            const char* subrange5 = 0,
            const char* subrange6 = 0
        );

        static void register_equivalent_descrs(
            const char* d1,
            const char* d2,
            const char* d3 = 0,
            const char* d4 = 0,
            const char* d5 = 0,
            const char* d6 = 0,
            const char* d7 = 0,
            const char* d8 = 0
        );
        
        static bool is_valid_subrange(
            IndexRange* parent,
            IndexRange* subrange
        );

        static bool is_threaded_compute();

        static void set_threaded_compute(bool flag);

        static char* malloc(size_t size);

        static void free(void* ptr, size_t size);

        static void print_memory_allocation();

        static bool print_cxn;

        static YetiOStream yetiout;

        static YetiRuntime::runtime_parallel_type_t get_parallel_type();

        static YetiMPIMessenger* get_mpi_messenger();

        static YetiMessenger* get_messenger();
};

IndexRange* index_range(const std::string& str);

template <typename T>
void
YetiOStream::print(const T& t)
{
    YetiRuntime::lock_print();
    std::cout << t;
    YetiRuntime::unlock_print();
}

template <class T>
YetiOStream&
operator<<(YetiOStream& os, T& t)
{
    os.template print(t);
    return os;
}

YetiOStream&
operator<<(YetiOStream& os, const std::string& str);

YetiOStream&
operator<<(YetiOStream& os, std::ostream& (*pf)(std::ostream&));


YetiOStream&
operator<<(YetiOStream& os, void* ptr);




}

#ifdef redefine_size_t
#undef size_t
#endif

#endif // Header Guard
