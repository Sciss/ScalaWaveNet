lazy val baseName  = "ScalaWaveNet"
lazy val baseNameL = baseName.toLowerCase

lazy val projectVersion = "0.1.0-SNAPSHOT"
lazy val mimaVersion    = "0.1.0"

lazy val buildSettings = Seq(
  name               := baseName,
  organization       := "de.sciss",
  version            := projectVersion,
  scalaVersion       := "2.11.11",
  crossScalaVersions := Seq("2.11.11", "2.10.6"),
  scalacOptions     ++= Seq("-deprecation", "-unchecked", "-feature", "-encoding", "utf8", "-Xfuture", "-Xlint"),
  licenses           := Seq("LGPL v2.1+" -> url("http://www.gnu.org/licenses/lgpl-2.1.txt")),
  homepage           := Some(url(s"https://github.com/Sciss/$baseName")),
  description        := "Experiments with audio files and artificial neural networks"
)

// ---- main dependencies ----

lazy val scalNetVersion   = "0.8.0"
lazy val audioFileVersion = "1.4.6"

// ---- test dependencies ----

lazy val standardSettings = buildSettings ++ Seq(
  libraryDependencies ++= Seq(
    "org.deeplearning4j" %% "scalnet"        % scalNetVersion,
    "de.sciss"           %% "scalaaudiofile" % audioFileVersion
  ),
  publishMavenStyle := true,
  publishArtifact in Test := false,
  publishTo :=
    Some(if (isSnapshot.value)
      "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"
    else
      "Sonatype Releases"  at "https://oss.sonatype.org/service/local/staging/deploy/maven2"
    ),
  pomExtra := {
    val n = baseName
    <scm>
      <url>git@github.com:Sciss/{n}.git</url>
      <connection>scm:git:git@github.com:Sciss/{n}.git</connection>
    </scm>
    <developers>
      <developer>
        <id>Sciss</id>
        <name>Hanns Holger Rutz</name>
        <url>http://github.com/Sciss</url>
      </developer>
    </developers>
  }
)

lazy val root = project.in(file("."))
  .settings(standardSettings)
  .settings(
    mimaPreviousArtifacts := Set("de.sciss" %% baseNameL % mimaVersion)
  )
