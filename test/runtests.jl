import cuTile, CUDA
using ParallelTestRunner

const init_code = quote
    using cuTile
    import cuTile as ct

    using FileCheck
end

testsuite = find_tests(pwd())

# Add examples to the test suite (requires workspaces, a Julia 1.12+ feature)
examples_root = joinpath(@__DIR__, "..", "examples")
if VERSION >= v"1.12"
    for (name, body) in find_tests(examples_root)
        path = joinpath(examples_root, name * ".jl")
        readline(path) == "# EXCLUDE FROM TESTING" && continue
        dir = dirname(path)
            testsuite["examples/$name"] = quote
            cd($dir) do
                project = Base.active_project()
                Base.set_active_project($dir)
                try
                    redirect_stdout(devnull) do
                        $body
                    end
                finally
                    Base.set_active_project(project)
                end
            end
        end
    end
end

# Only include executing tests when CUDA is functional
args = parse_args(ARGS)
if filter_tests!(testsuite, args)
    cuda_functional = CUDA.functional()
    filter!(testsuite) do (test, _)
        if startswith(test, "device/") || startswith(test, "host/") || startswith(test, "examples/")
            return cuda_functional
        else
            return true
        end
    end
end

runtests(cuTile, ARGS; init_code, testsuite)
