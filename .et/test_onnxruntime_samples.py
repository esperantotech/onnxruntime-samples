import os
import pytest
import shutil

from run_wrapper import run


# Base project and build directories for models
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ORT_PROJECT_ROOT = os.path.join(PROJECT_ROOT, "..")  # assuming onnxruntime-samples is a subfolder of onnxruntime
BASE_SOURCE_PATH = os.path.join(PROJECT_ROOT, "models")
BASE_BUILD_PATH  = os.path.join(PROJECT_ROOT, "builds")
ET_SDK_HOME = os.getenv("ET_SDK_HOME")  # Detect Docker environment if present


def get_cmake_toolchain_file(family):
    if ET_SDK_HOME:
        cmake_toolchain_file = f"{ET_SDK_HOME}/.builds/host/conan_toolchain.cmake"
        print(f"Detected Docker environment. Using ET_SDK_HOME: {ET_SDK_HOME}")
    else:
        build_path = os.path.join(PROJECT_ROOT, "..", "build", "Release")
        cmake_toolchain_file = f"{build_path}/generators/conan_toolchain.cmake"
        print(f"Detected developer environment. Using Conan setup for family '{family}'.")
    return cmake_toolchain_file


@pytest.fixture(scope="session")
def cxx_compile(request):
    """Compile C++ binaries for a given family, using the structured directories."""
    family = request.param  # Access the family parameter, e.g., "image-classifier" or "language-model"
    source_path = os.path.join(BASE_SOURCE_PATH, family, "c++")
    build_path = os.path.join(BASE_BUILD_PATH, family, "c++")

    # Ensure source path with CMakeLists.txt exists
    if not os.path.exists(os.path.join(source_path, "CMakeLists.txt")):
        raise FileNotFoundError(f"CMakeLists.txt not found at expected path {source_path}")

    cmake_toolchain_file = get_cmake_toolchain_file(family)

    # Compile the binaries using the source and build paths
    configure_cmd = (
        f"cmake -S {source_path} -B {build_path} "
        f"-DCMAKE_TOOLCHAIN_FILE={cmake_toolchain_file} "
        f"-DCMAKE_BUILD_TYPE=Release "
        "-G Ninja"
    )
    build_cmd = f"cmake --build {build_path}"

    print(f"Running {configure_cmd}")
    run(configure_cmd)
    print(f"Running {build_cmd}")
    run(build_cmd)
    print(f"C++ binaries for {family} compiled successfully.")

    return build_path  # Return build path for test use



@pytest.fixture(scope="session", autouse=True)
def python_module_req(request):
    """Automatically setups python module requirements."""
    def symlink_force(src, dst):
        """Create a symlink, removing the destination if it exists."""
        try:
            # Check if the destination already exists
            if os.path.exists(dst):
                os.remove(dst)  # Remove existing symlink or file

            # Create a symlink from src to dst
            os.symlink(src, dst)
            print(f"Symlink created: {dst} -> {src}")
        except OSError as e:
            print(f"Failed to create symlink from {src} to {dst}: {e}")

    test_families = ["image-classifier", "language-model", "text-to-image"]
    if "onnxruntime" not in os.environ.get("PYTHONPATH", ""):
        # create symlinks to 'onnxruntime' under test
        for family in test_families:

            symlink_src = os.path.join(ORT_PROJECT_ROOT, "build", "Release", "onnxruntime")
            symlink_dst = os.path.join(BASE_SOURCE_PATH, family, "python", "onnxruntime")
            if os.path.exists(symlink_src):
                symlink_force(symlink_src, symlink_dst)
            else:
                print(f"Source directory does not exist: {symlink_src}")

@pytest.fixture(scope="session")
def generate_squad_datasets(request):
    """This fixture will generate datasets required for bert"""
    seq_len = 128
    models = ["bert", "albert", "distilbert"]
    for model in models:
        output_dir = f"DownloadArtifactory/input_tensors/{model}_squad_{seq_len}"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        run(f"artifacts/squad-dataset-to-input-tensor.py --model {model} --seq-length {seq_len} --batch-size 1 --output-dir {output_dir} --file DownloadArtifactory/datasets/squadV1.1/data/squad-v1.1-dev.json")


def run_cxx_sample(request, cmd, should_succeed=True):
    result = run(cmd, output_path='tests/' + request.node.name)
    if should_succeed:
        if result.returncode != 0:
            assert False, f"Execution failed with return code {result.returncode} failed"
        else:
            print('Execution finished successfully')
    else:
        if result.returncode < 0:
            assert False, f'Execution was expected to fail gracefully, but finished with code {result.returncode}'
        elif result.returncode == 0:
            assert False, f'Execution was expected to fail gracefully, but finished successfully'
        else:
            print(f'Execution failed as expected with return code {result.returncode}')


def run_py_sample(request, test_family, test_module,
                  test_model=None, num_tokens=None, batch=None, launches=None, run_mode=None,
                  with_tracing=None, with_warmup=None, with_performance=None, with_input=None,
                  bert_variant=None,
                  should_succeed=True):
    if test_model is not None:  # llm(s)
        m_param = f'-m DownloadArtifactory/models/{test_model}/model.onnx'
    elif run_mode is not None:  # image-classifiers
        m_param = f'-m {run_mode}'
    else:
        m_param = ''
    t_param = f"--tokenizer DownloadArtifactory/tokenizer/{test_model.split('-')[0]}" if test_model is not None else ''
    num_tokens_param = f'-g {num_tokens}' if num_tokens else ''
    warmup_param = '--warm-up' if with_warmup else ''
    tracing_param = '--enable-tracing' if with_tracing else ''
    batch_param = f'--batch {batch}' if batch is not None else ''
    launches_param = f'--launches {launches}' if launches is not None else ''
    performance_param = f'--performance' if with_performance else ''
    input_param = f'-i {with_input}' if with_input else ''
    bert_variant_param = f"--bert-variant {bert_variant}" if bert_variant else ''
    artifacts_param = "--artifacts DownloadArtifactory" if test_model is None else ""
    result = run(
        f"python3 models/{test_family}/python/{test_module} "
        f"{m_param} {t_param} {num_tokens_param} {warmup_param} {tracing_param} {batch_param} {launches_param} {performance_param} {input_param} {bert_variant_param} "
        f"{artifacts_param}",
        output_path='tests/' + request.node.name
    )
    if should_succeed:
        if result.returncode != 0:
            assert False, f"Execution failed with return code {result.returncode} failed"
        else:
            print('Execution finished successfully')
    else:
        if result.returncode < 0:
            assert False, f'Execution was expected to fail gracefully, but finished with code {result.returncode}'
        elif result.returncode == 0:
            assert False, f'Execution was expected to fail gracefully, but finished successfully'
        else:
            print(f'Execution failed as expected with return code {result.returncode}')


@pytest.mark.cxx
@pytest.mark.ic
class TestImageClassifiersCxx:
    family = "image-classifier"
    test_launches = [1, 1000, pytest.param(10000, marks=pytest.mark.long)]

    @pytest.mark.parametrize("cxx_compile", ["image-classifier"], indirect=True)
    def test_mnist(self, cxx_compile, request):
        """Test mnist"""
        binary_path = f"{cxx_compile}/mnist --artifact_folder DownloadArtifactory"
        run_cxx_sample(request, binary_path)

    # Test for language-model Python binaries
    @pytest.mark.cxx
    @pytest.mark.parametrize("cxx_compile", ["image-classifier"], indirect=True)
    @pytest.mark.parametrize('batch', [None, 1, 2, 4], ids=["", "batch_1", "batch_2", "batch_4"])
    @pytest.mark.parametrize('launches', test_launches)
    @pytest.mark.parametrize('run_mode', ["sync", "async"])
    @pytest.mark.parametrize('with_iobinding', [True, False], ids=["with_iobindings", "without_iobindings"])
    def test_resnet50(self, cxx_compile, batch, launches, run_mode, with_iobinding, request):
        """Test resnet50"""
        if with_iobinding:
            pytest.skip("unsupported configuration")

        batch_cmd = f"-batchSize {batch}" if batch is not None else ""
        launches_cmd = f"-totalInferences {launches}"
        run_mode_cmd = "-asyncMode" if run_mode == "async" else ""
        iobinding_cmd = "-iobinding" if with_iobinding else ""
        binary_path = (f"{cxx_compile}/resnet --artifact_folder DownloadArtifactory "
                       f"{batch_cmd} {launches_cmd} {run_mode_cmd} {iobinding_cmd}")
        run_cxx_sample(request, binary_path)




@pytest.mark.python
@pytest.mark.ic
class TestImageClassifiersPython:
    family = "image-classifier"
    test_launches = [1, 10, pytest.param(100, marks=pytest.mark.long)]

    @pytest.mark.parametrize('with_tracing', [True, False], ids=["with_tracing","without_tracing"])
    def test_mnist(self, with_tracing, request):
        """Test mnist.py"""
        run_py_sample(request,
                      self.family, "mnist.py",
                      with_tracing=with_tracing)

    @pytest.mark.parametrize('batch', [1, 2, 4], ids=["batch_1", "batch_2", "batch_4"])
    @pytest.mark.parametrize('launches', test_launches)
    @pytest.mark.parametrize('run_mode', ["sync", "async"])
    @pytest.mark.parametrize('with_tracing', [True, False], ids=["with_tracing","without_tracing"])
    @pytest.mark.parametrize('with_warmup', [True, False], ids=["with_warmup","without_warmup"])
    def test_resnet50(self, batch, launches, run_mode, with_tracing, with_warmup, request):
        """Test resnet50.py"""
        run_py_sample(request,
                      self.family, "resnet50.py",
                      batch=batch,
                      launches=launches,
                      run_mode=run_mode,
                      with_tracing=with_tracing,
                      with_warmup=with_warmup)

    @pytest.mark.parametrize('batch', [1, 2, 4], ids=["batch_1", "batch_2", "batch_4"])
    @pytest.mark.parametrize('launches', test_launches)
    @pytest.mark.parametrize('run_mode', ["sync", "async"])
    @pytest.mark.parametrize('with_tracing', [True, False], ids=["with_tracing","without_tracing"])
    @pytest.mark.parametrize('with_warmup', [True, False], ids=["with_warmup","without_warmup"])
    def test_vgg16(self, batch, launches, run_mode, with_tracing, with_warmup, request):
        """Test vgg16.py & vgg19.py"""
        run_py_sample(request,
                      self.family, f"vgg16.py",
                      batch=batch,
                      launches=launches,
                      run_mode=run_mode,
                      with_tracing=with_tracing,
                      with_warmup=with_warmup)

    @pytest.mark.parametrize('batch', [1], ids=["batch_1"]) # vgg19 only accepts batch 1
    @pytest.mark.parametrize('launches', test_launches)
    @pytest.mark.parametrize('run_mode', ["sync", "async"])
    @pytest.mark.parametrize('with_tracing', [True, False], ids=["with_tracing","without_tracing"])
    @pytest.mark.parametrize('with_warmup', [True, False], ids=["with_warmup","without_warmup"])
    def test_vgg19(self, batch, launches, run_mode, with_tracing, with_warmup, request):
        """Test vgg16.py & vgg19.py"""
        run_py_sample(request,
                      self.family, f"vgg19.py",
                      batch=batch,
                      launches=launches,
                      run_mode=run_mode,
                      with_tracing=with_tracing,
                      with_warmup=with_warmup)


    @pytest.mark.parametrize('batch', [1, 2, 4], ids=["batch_1", "batch_2", "batch_4"])
    @pytest.mark.parametrize('launches', test_launches)
    @pytest.mark.parametrize('run_mode', ["sync", "async"])
    @pytest.mark.parametrize('with_tracing', [True, False], ids=["with_tracing","without_tracing"])
    @pytest.mark.parametrize('with_warmup', [True, False], ids=["with_warmup","without_warmup"])
    @pytest.mark.parametrize('with_performance', [True, False], ids=["with_performance","without_performance"]) # mobilenet specific
    def test_mobilenet(self, batch, launches, run_mode, with_tracing, with_warmup, with_performance, request):
        """Test mobilenet.py"""
        run_py_sample(request,
                      self.family, "mobilenet.py",
                      batch=batch,
                      launches=launches,
                      run_mode=run_mode,
                      with_tracing=with_tracing,
                      with_warmup=with_warmup,
                      with_performance=with_performance)


@pytest.mark.python
@pytest.mark.lm
class TestLanguageModelsPython:
    family = "language-model"
    test_num_tokens = [10, pytest.param(30, marks=pytest.mark.long)]

    @pytest.mark.parametrize('bert_variant', ["bert", "bert-large", "albert", "distilbert"])
    def test_bert(self, bert_variant, generate_squad_datasets, request):
        """Test bert.py"""
        run_py_sample(request, self.family, "bert.py", bert_variant=bert_variant)

    test_models = [
        'vicuna-1.5-7b-kvc-int4',
        'mistral-instruct-7b-kvc-fp16',
        'llama3-8b-instruct-kvc-int4',
        pytest.param('llama3-8b-instruct-kvc-fp16', marks=pytest.mark.long)
    ]
    test_batch = [1, pytest.param(2, marks=pytest.mark.long), pytest.param(4, marks=pytest.mark.long)]

    @pytest.mark.parametrize('with_tracing', [True, False], ids=["with_tracing","without_tracing"])
    @pytest.mark.parametrize('num_tokens', test_num_tokens)
    @pytest.mark.parametrize('batch', test_batch, ids=["batch_1", "batch_2", "batch_4"])
    @pytest.mark.parametrize('model', test_models)
    def test_llm_kvc(self, model, num_tokens, batch, with_tracing, request):
        """Test llm-kvc.py"""
        run_py_sample(request,
                      self.family, "llm-kvc.py",
                      test_model=model,
                      num_tokens=num_tokens,
                      batch=batch,
                      with_tracing=with_tracing)

    def test_llava(self, batch, with_tracing, request):
        """Test llava.py"""
        run_py_sample(request,
                      self.family, "llava.py",
                      batch=batch,
                      with_tracing=with_tracing,
                      with_input="artifacts/doge.jpg")