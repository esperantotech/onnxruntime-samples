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
                  test_model=None, num_tokens=None, new_tokens=None, batch=None, launches=None, run_mode=None,
                  with_tracing=None, with_warmup=None, with_input=None, with_output=None, with_golden=None,
                  bert_variant=None, fp16=None, precision=None, provider=None,
                  image=None, expected_result=None,
                  prompt=None, artifacts_dir=True,
                  verbose=True,
                  should_succeed=True):
    if test_model is not None:  # llm(s)
        m_param = f'-m DownloadArtifactory/models/{test_model}/model.onnx'
    elif run_mode is not None:  # image-classifiers
        m_param = f'-m {run_mode}'
    else:
        m_param = ''
    t_param = f"--tokenizer DownloadArtifactory/tokenizer/{test_model.split('-')[0]}" if test_model is not None else ''
    num_tokens_param = f'--generate-tokens {num_tokens}' if num_tokens else ''
    new_tokens_param = f'--new-tokens {new_tokens}' if new_tokens else ''
    warmup_param = '--warm-up' if with_warmup else ''
    tracing_param = '--enable-tracing' if with_tracing else ''
    batch_param = f'--batch {batch}' if batch is not None else ''
    launches_param = f'--launches {launches}' if launches is not None else ''
    input_param = f'-i {with_input}' if with_input else ''
    output_param = f"-o {with_output}" if with_output else ''
    golden_param = f"--golden {with_golden}" if with_golden else ''
    bert_variant_param = f"--bert-variant {bert_variant}" if bert_variant else ''
    fp16_param = f"--fp16" if fp16 else ''
    precision_param = f"--precision {precision}" if precision is not None else ''
    provider_param = f"--provider {provider}" if provider is not None else ''
    prompt_param = f"--prompt \"{prompt}\"" if prompt is not None else ''
    image = f"--image {image}" if image else ''
    expected_result = f"--expected-result {expected_result}" if expected_result else ''
    artifacts_param = "--artifacts DownloadArtifactory" if artifacts_dir and test_model is None else ""
    verbose_param = "-v" if verbose else ""
    result = run(
        f"python3 models/{test_family}/python/{test_module} "
        f"{m_param} {t_param} {num_tokens_param} {new_tokens_param} {warmup_param} {tracing_param} {batch_param} "
        f"{launches_param} {input_param} {output_param} {bert_variant_param} {fp16_param} {precision_param} "
        f"{provider_param} {golden_param} {prompt_param} "
        f"{image} {expected_result} "
        f"{artifacts_param} {verbose_param}",
        output_path='tests/' + request.node.name
    )
    if should_succeed:
        if result.returncode != 0:
            assert False, f"Execution failed with return code {result.returncode} failed stdout {result.stdout} stderr {result.stderr}"
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
    test_launches = [1, 1000, 10000, pytest.param(100000, marks=pytest.mark.long)]

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
    test_launches = [1, 10, 100, pytest.param(1000, marks=pytest.mark.long)]

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
    def test_mobilenet(self, batch, launches, run_mode, with_tracing, with_warmup, request):
        """Test mobilenet.py"""
        run_py_sample(request,
                      self.family, "mobilenet.py",
                      batch=batch,
                      launches=launches,
                      run_mode=run_mode,
                      with_tracing=with_tracing,
                      with_warmup=with_warmup)

    @pytest.mark.parametrize('image', ['cats.jpg'])
    @pytest.mark.parametrize('expected_result', ["cat cat remote remote"])
    def test_yolo_v8(self, image, expected_result, request):
        """Test yolo_v8.py"""
        run_py_sample(request,
                      self.family, "yolo_v8.py",
                      image=f'artifacts/{image}',
                      expected_result=expected_result,
                      )

    def test_retinanet(self, request):
        """Test retinanet.py"""
        run_py_sample(request,
                      self.family, "retinanet.py",
                      )

    def test_transunet(self, request):
        """Test transunet.py"""
        run_py_sample(request,
                      self.family, "transunet.py",
                      )

    def test_unet(self, request):
        """Test unet.py"""
        run_py_sample(request,
                      self.family, "unet.py",
                      )


@pytest.mark.python
@pytest.mark.lm
class TestLanguageModelsPython:
    family = "language-model"
    test_batch = [1, 2, pytest.param(4, marks=pytest.mark.long)]
    test_launches = [1, 10, pytest.param(100, marks=pytest.mark.long)]

    @pytest.mark.parametrize('fp16', [True, False], ids=["fp16","fp32"])
    @pytest.mark.parametrize('batch', test_batch, ids=["batch_1", "batch_2", "batch_4"])
    @pytest.mark.parametrize('launches', test_launches)
    @pytest.mark.parametrize('with_warmup', [True, False], ids=["with_warmup","without_warmup"])
    @pytest.mark.parametrize('bert_variant', ["bert", "bert-large", "albert", "distilbert"])
    def test_bert(self, batch, launches, fp16, with_warmup, bert_variant, generate_squad_datasets, request):
        """Test bert.py"""
        run_py_sample(request, self.family, "bert.py",
                      fp16=fp16,
                      batch=batch,
                      launches=launches,
                      with_warmup=with_warmup,
                      bert_variant=bert_variant)

    test_num_tokens = [10, pytest.param(30, marks=pytest.mark.long)]
    test_models = [
        'vicuna-7b-v1.5-kvc-AWQ-int4-onnx',
        'mistral-7b-Instruct-v0.2-kvc-fp16-onnx',
        'llama3-8b-Instruct-kvc-AWQ-int4-onnx',
        pytest.param('llama3-8b-Instruct-kvc-fp16-onnx', marks=pytest.mark.long)
    ]

    @pytest.mark.parametrize('with_tracing', [True, False], ids=["with_tracing","without_tracing"])
    @pytest.mark.parametrize('num_tokens', test_num_tokens)
    @pytest.mark.parametrize('batch', test_batch, ids=["batch_1", "batch_2", "batch_4"])
    @pytest.mark.parametrize('model', test_models)
    def test_llm_kvc(self, model, num_tokens, batch, with_tracing, request):
        """Test llm-kvc.py"""
        if "vicuna" in model and batch > 1:
            pytest.xfail("vicuna with batch!=1 is known to crash")
        run_py_sample(request,
                      self.family, "llm-kvc.py",
                      test_model=model,
                      num_tokens=num_tokens,
                      batch=batch,
                      with_tracing=with_tracing)

    @pytest.mark.parametrize('input', ["1984.m4a"])
    @pytest.mark.parametrize('with_tracing', [True, False], ids=["with_tracing","without_tracing"])
    @pytest.mark.parametrize('num_tokens', test_num_tokens)
    def test_whisper_kvc(self, input, num_tokens, with_tracing, request):
        """Test whisper-kvc.py"""
        run_py_sample(request,
                      self.family, "whisper-kvc.py",
                      new_tokens=num_tokens,
                      with_tracing=with_tracing,
                      with_input=f"artifacts/{input}")

    @pytest.mark.parametrize('input', ["doge.jpg"])
    @pytest.mark.parametrize('with_tracing', [True, False], ids=["with_tracing","without_tracing"])
    @pytest.mark.parametrize('num_tokens', test_num_tokens)
    def test_llava(self, input, num_tokens, with_tracing, request):
        """Test llava.py"""
        run_py_sample(request,
                      self.family, "llava.py",
                      new_tokens=num_tokens,
                      with_tracing=with_tracing,
                      with_input=f"artifacts/{input}")

@pytest.mark.python
@pytest.mark.t2i
class TestText2ImageModelsPython:
    family = "text-to-image"

    @pytest.mark.parametrize('precision', ["fp32", "fp16"])
    def test_openjourney(self, precision, request):
        """Test openjourney.py"""
        golden_cpu = f"golden_image_cpu_{precision}.png"
        image_et = f"test_image_etglow_{precision}.png"
        run_py_sample(request, self.family, "openjourney.py",
                      prompt="foggy lake under waterfall with full moon",
                      precision=precision,
                      provider="cpu",
                      with_output=golden_cpu)
        run_py_sample(request, self.family, "openjourney.py",
                      prompt="foggy lake under waterfall with full moon",
                      precision=precision,
                      provider="etglow",
                      with_output=image_et)
        run_py_sample(request, self.family, "testimages.py",
                      precision=precision,
                      with_input=image_et,
                      with_golden=golden_cpu,
                      artifacts_dir=False)
